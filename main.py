import os
print("STARTING SCRIPT")
import torch
import numpy as np
import json
from luxai_s2.env import LuxAI_S2
from env_wrapper import LuxS2Wrapper
from logger import TrajectoryLogger
from agent import OfflineCQLAgent


# ─── JSON Encoder for numpy types ─────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


# ─── Global State for Coordination ──────────────────────────────────────────
UNIT_HOME_MAP = {}         # robot_id -> factory_id
GLOBAL_PENDING_SPAWNS = {"player_0": [], "player_1": []} # current turn's [x, y] spawn locations
UNIT_ROLE_MAP = {}         # robot_id -> "ice" or "ore"
UNIT_STUCK_TIME = {}
UNIT_PREV_POS = {}
GLOBAL_HUB_BUILT = {"player_0": False, "player_1": False}
GLOBAL_RESERVED_TILES = set()

# ─── Heuristic ────────────────────────────────────────────────────────────────
def get_heuristic_actions(obs, player, env):
    """
    env_steps == 0  → Bidding  (AlphaStrike, bid=0)
    real_steps < 0  → Placement (fabrika kur)
    real_steps >= 0 → Normal oyun
    """
    p_obs = obs.get(player, {})
    if not p_obs:
        return {}

    env_steps  = env.state.env_steps
    real_steps = env.state.real_env_steps

    # 1. Bidding — yalnızca env_steps==0
    if env_steps == 0:
        return dict(bid=0, faction="AlphaStrike")

    # 2. Factory Placement — real_steps < 0 (env_steps > 0)
    if real_steps < 0:
        # factories_to_place top-level'da değil, teams içinde!
        team_data          = p_obs.get("teams", {}).get(player, {})
        factories_to_place = int(team_data.get("factories_to_place", 0))

        if factories_to_place > 0:
            board       = p_obs.get("board", {})
            spawns_mask = board.get("valid_spawns_mask")

            if spawns_mask is not None:
                valid_spawns = np.argwhere(np.array(spawns_mask) == 1)
                if len(valid_spawns) > 0:
                    # Zorunlu Mesafe Kontrolü: Kurulu fabrikalar + bu tur kurulacak (Pending) olanlar
                    existing_factories = []
                    for t_id, f_dict in env.state.factories.items():
                        for fid, f in f_dict.items():
                            ef_pos = f.pos.pos if hasattr(f.pos, "pos") else f.pos
                            existing_factories.append(np.array(ef_pos))
                    
                    # Bu tur diğer oyuncunun seçtiği noktaları da ekle
                    for px_lists in GLOBAL_PENDING_SPAWNS.values():
                        for px in px_lists:
                            existing_factories.append(np.array(px))
                            
                    if existing_factories: # Artık hem P0 hem P1 için mesafe kontrolü yapıyoruz
                        dists_to_others = np.array([
                            np.min([np.abs(s - ef).sum() for ef in existing_factories])
                            for s in valid_spawns
                        ])
                        mask = (dists_to_others >= 5) & (dists_to_others <= 15)
                        if np.any(mask):
                            valid_spawns = valid_spawns[mask]
                        else:
                            mask_fallback = dists_to_others >= 5
                            if np.any(mask_fallback):
                                valid_spawns = valid_spawns[mask_fallback]

                    # Seçilen hedefe (ice) en yakın olanı bul
                    res_pref = "ice"
                    target_map = board.get(res_pref)
                    if target_map is not None:
                        coords = np.argwhere(np.array(target_map) > 0)
                        if len(coords) > 0:
                            dists = np.array([
                                np.min(np.abs(coords - s).sum(axis=1))
                                for s in valid_spawns
                            ])
                            spawn_xy = [int(valid_spawns[np.argmin(dists)][0]),
                                        int(valid_spawns[np.argmin(dists)][1])]
                        else:
                            np.random.shuffle(valid_spawns)
                            spawn_xy = [int(valid_spawns[0][0]), int(valid_spawns[0][1])]
                    else:
                        np.random.shuffle(valid_spawns)
                        spawn_xy = [int(valid_spawns[0][0]), int(valid_spawns[0][1])]

                    avail_water = int(team_data.get("water", 150))
                    avail_metal = int(team_data.get("metal", 150))

                    # Kural Ihlali ve S2 Motor Cokmelerine karsi Sabit Strateji: S2 motoru 0/0 kurbanlarini
                    # veya asiri dar alana (safe_spawns) rastgele fabrikalari sikistirinca Player 1'i diskalifiye edip
                    # 16. adimda maci bitiriyor. O yuzden tum fabrikalara, kalan havuzu auto-balance yapmasi 
                    # icin per_water/per_metal deklare ediyoruz ki sistem cokmesin ve 1000+ adim yasayalim.
                    per_water   = avail_water
                    per_metal   = avail_metal
                    if not GLOBAL_HUB_BUILT.get(player, False):
                        print(f"  [PLACEMENT] {player}: MEGA-HUB spawn={spawn_xy} water={per_water} metal={per_metal}")
                        GLOBAL_HUB_BUILT[player] = True
                    return dict(spawn=spawn_xy, metal=per_metal, water=per_water)
            else:
                print(f"  [ERROR] {player}: valid_spawns_mask missing from board!")
        return {}

    # 3. Normal Oyun
    actions   = {}
    units     = p_obs.get("units", {}).get(player, {})
    factories = p_obs.get("factories", {}).get(player, {})
    board     = p_obs.get("board", {})

    if not board:
        return {}

    ice = np.array(board.get("ice", [[0]]))
    ore = np.array(board.get("ore", [[0]]))
    rubble = np.array(board.get("rubble", [[0]]))
    
    ice_coords = np.argwhere(ice > 0)
    ore_coords = np.argwhere(ore > 0)
    general_resource_coords = np.argwhere((ice > 0) | (ore > 0))
    if len(ice_coords) == 0: ice_coords = general_resource_coords
    if len(ore_coords) == 0: ore_coords = general_resource_coords

    # --- KOORDİNASYON FAZI: Rescue (Pull) Kontrolü ---
    thirsty_factories = []
    for f_id, f_data in factories.items():
        if f_data.get("cargo", {}).get("water", 0) < 30:
            thirsty_factories.append({"id": f_id, "pos": f_data["pos"], "water": f_data.get("cargo", {}).get("water", 0)})
            
    # Rescue görevlendirmeleri (Geçici)
    rescue_assignments = {} # unit_id -> target_factory_pos
    
    # 30'un altındaki fabrikalar için robot rezerve et
    if len(thirsty_factories) > 0 and len(units) > 0:
        # Sadece buz taşıyan robotlar kurtarıcı olabilir
        helpers_with_ice = []
        for uid, udata in units.items():
            if udata.get("cargo", {}).get("ice", 0) > 0:
                # Kendi evi de susuzsa başkasına yardım edemez (Override kuralı hazırlığı)
                home_fid = UNIT_HOME_MAP.get(uid)
                home_water = factories.get(home_fid, {}).get("cargo", {}).get("water", 999) if home_fid else 999
                if home_water >= 30:
                    helpers_with_ice.append({"id": uid, "pos": udata["pos"], "ice": udata["cargo"]["ice"]})
        
        for thirsty in thirsty_factories:
            if not helpers_with_ice: break
            
            # Mesafeye göre sırala
            helpers_with_ice.sort(key=lambda h: np.abs(np.array(h["pos"]) - np.array(thirsty["pos"])).sum())
            
            # En yakın robotun yükü >= 280 buz ise sadece o gider
            closest = helpers_with_ice.pop(0)
            rescue_assignments[closest["id"]] = thirsty["pos"]
            
            if closest["ice"] < 280 and helpers_with_ice:
                # Takviye: İkinci robotu da çağır
                second = helpers_with_ice.pop(0)
                rescue_assignments[second["id"]] = thirsty["pos"]
    # -----------------------------------------------

    # ENEMY FACTORY / AVOIDANCE BUFFER (Pathfinding Engel Haritası)
    enemy = "player_1" if player == "player_0" else "player_0"
    enemy_factories = p_obs.get("factories", {}).get(enemy, {})
    enemy_factory_tiles = set()
    for f_id, f_data in enemy_factories.items():
        fx, fy = f_data["pos"]
        # Buffer zone incl. Chebyshev <= 2 (5x5 footprint for 1-tile safe separation)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                enemy_factory_tiles.add((fx + dx, fy + dy))

    def get_safe_direction(u_pos, t_pos, obstacles):
        diff = t_pos - u_pos
        dirs = []
        if abs(diff[0]) > abs(diff[1]):
            dirs.extend([(2 if diff[0] > 0 else 4), (3 if diff[1] > 0 else 1), (1 if diff[1] > 0 else 3), (4 if diff[0] > 0 else 2)])
        else:
            dirs.extend([(3 if diff[1] > 0 else 1), (2 if diff[0] > 0 else 4), (4 if diff[0] > 0 else 2), (1 if diff[1] > 0 else 3)])
            
        move_deltas = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
        for d in dirs:
            nx, ny = u_pos[0] + move_deltas[d][0], u_pos[1] + move_deltas[d][1]
            if (nx, ny) not in obstacles:
                return d
        return 0 # Return 0 (center/wait) if trapped

    for unit_id, unit_data in units.items():
        pos = np.array(unit_data["pos"])
        cargo = unit_data.get("cargo", {})
        total_cargo = cargo.get("ice", 0) + cargo.get("ore", 0)
        unit_power = unit_data.get("power", 0)
        has_empty_queue = (len(unit_data.get("action_queue", [])) == 0)

        # --- HOME FACTORY ATAMASI ---
        if unit_id not in UNIT_HOME_MAP and len(factories) > 0:
            f_ids = list(factories.keys())
            f_positions = [factories[fid]["pos"] for fid in f_ids]
            closest_f_idx = np.argmin(np.abs(np.array(f_positions) - pos).sum(axis=1))
            UNIT_HOME_MAP[unit_id] = f_ids[closest_f_idx]
        
        home_fid = UNIT_HOME_MAP.get(unit_id)
        home_data = factories.get(home_fid, {})
        home_pos = home_data.get("pos")
        home_water = home_data.get("cargo", {}).get("water", 999)

        # EMERGENCY OVERRIDE CHECK
        is_emergency = False
        if unit_power < 150 and home_pos is not None:
            is_emergency = True
            
        # Stuck time tracking (Enerjiyi bosuna tuketmemek icin)
        prev_pos = UNIT_PREV_POS.get(unit_id, pos)
        if np.array_equal(pos, prev_pos):
            UNIT_STUCK_TIME[unit_id] = UNIT_STUCK_TIME.get(unit_id, 0) + 1
        else:
            UNIT_STUCK_TIME[unit_id] = 0
        UNIT_PREV_POS[unit_id] = pos

        # Kilitlenmeyi onle: Queue guncellenmesi 1 guc yer!
        q_action = unit_data.get("action_queue")[0][0] if not has_empty_queue else -1
        is_on_home_anyway = (home_pos is not None) and max(abs(pos[0] - home_pos[0]), abs(pos[1] - home_pos[1])) <= 1
        
        # Kullanıcı İsteği ECO 02 (Factory Digging Deadlock):
        if is_on_home_anyway and q_action == 3: # Fabrika zeminini kazıyorsa
            actions[unit_id] = [] # Vandalizmi durdur, kuyruğu temizle (Clear Queue)
            has_empty_queue = True
            
        # INVALID ACTION QUEUE CLEARANCE (Rakip fabrikayı süpürme)
        if not has_empty_queue and q_action == 0:
            q_dir = unit_data.get("action_queue")[0][1]
            move_deltas = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0), 0: (0, 0)}
            nx = pos[0] + move_deltas.get(q_dir, (0,0))[0]
            ny = pos[1] + move_deltas.get(q_dir, (0,0))[1]
            if (nx, ny) in enemy_factory_tiles:
                actions[unit_id] = []
                has_empty_queue = True
            
        # Ayrıca fabrikaya her girdiğinde hafızayı tazelemek (PickUp / Transfer taze emri için kilidi kırma)
        if is_on_home_anyway and not has_empty_queue and q_action not in [1, 2]:
            has_empty_queue = True # Bu sahte "Boş" bayrağı, is_emergency false olsa bile blok atlamamasını sağlar!

        if is_emergency and not has_empty_queue:
            if q_action in [0, 2]:
                continue
        
        # Eğer queue doluysa ve acil durum / fabrika reset'i yoksa, bu robotu atla!
        if not has_empty_queue and not is_emergency:
            continue

        # --- ROL ATAMASI (Dinamik Ekonomi) ---
        if unit_id not in UNIT_ROLE_MAP:
            # %70 Ice, %30 Ore olarak rolleri dağıt
            UNIT_ROLE_MAP[unit_id] = "ice" if np.random.rand() < 0.70 else "ore"
        
        my_role = UNIT_ROLE_MAP[unit_id]

        # --- GÖREV HİYERARŞİSİ ---
        target_pos = None
        is_commited = False # Eve dönüyor mu/Yardıma mı gidiyor?

        # 0. LOW POWER EMERGENCY: Gücüm 150'nin altındaysa direkt eve dön (Şarj olmaya)
        if is_emergency:
            target_pos = home_pos
            is_commited = True

        # 1. OVERRIDE: Kendi evim yanıyorsa her şeyi bırak geri dön!
        elif home_water < 30 and total_cargo > 0:
            target_pos = home_pos
            is_commited = True
        
        # 2. RESCUE (PULL): Eğer kurtarıcı olarak atandıysam (ve evim güvendeyse)
        elif unit_id in rescue_assignments:
            target_pos = rescue_assignments[unit_id]
            is_commited = True
        
        # 3. HOME RETURN (PUSH): Yüküm dolduysa sadece kendi evime dön
        elif total_cargo >= 400 and home_pos is not None:
            target_pos = home_pos
            is_commited = True

        if is_commited and target_pos is not None:
            # 3x3 Fabrika Alanı Chebyshev Checker: Robot merkeze en fazla 1 uzaklıkta mı?
            is_on_target = max(abs(pos[0] - target_pos[0]), abs(pos[1] - target_pos[1])) <= 1
            
            if is_on_target:
                # Sadece kendi evimizdeysek şarj olabiliriz
                is_on_home = (home_pos is not None) and max(abs(pos[0] - home_pos[0]), abs(pos[1] - home_pos[1])) <= 1
                
                action_assigned = False

                if is_on_home:
                    home_fac_power = home_data.get("power", 0)
                    if unit_power < 3000 and home_fac_power > 0: # 1500 baraji kalkti, 3000'e kadar sarj izni
                        pickup_amount = min(home_fac_power, 3000 - unit_power)
                        if pickup_amount > 0:
                            actions[unit_id] = [np.array([2, 0, 4, pickup_amount, 0, 1])]
                            action_assigned = True

                # Kargo Aktarımı (Target ne olursa olsun yapilabilir, ornegin rescue edilen fabrikaya water atmak icin)
                if not action_assigned and total_cargo > 0:
                    if cargo.get("ice", 0) > 0:
                        actions[unit_id] = [np.array([1, 0, 0, cargo["ice"], 0, 1])] # Ice transfer
                        action_assigned = True
                    elif cargo.get("ore", 0) > 0:
                        actions[unit_id] = [np.array([1, 0, 1, cargo["ore"], 0, 1])] # Ore transfer
                        action_assigned = True

                # EJECTION PROTOCOL: Kargomuz yok, şarjımız 1500+ ve EVDEYSEK (hemen cikip Idle Task'a düs!)
                if not action_assigned and is_on_home and unit_power > 1500 and total_cargo == 0 and len(ice_coords) > 0:
                    dist_to_ice = np.abs(ice_coords - pos).sum(axis=1)
                    closest_ice = ice_coords[np.argmin(dist_to_ice)]
                    diff_ice = closest_ice - pos
                    direction = get_safe_direction(pos, closest_ice, enemy_factory_tiles)
                    actions[unit_id] = [np.array([0, direction, 0, 0, 0, 1])]
                    action_assigned = True
                    
                if action_assigned:
                    continue 
                else:
                    is_commited = False # Action: None 'durumuna pusu kurmak yerine is bulmaya gec!

            else:
                direction = get_safe_direction(pos, target_pos, enemy_factory_tiles)
                actions[unit_id] = [np.array([0, direction, 0, 0, 0, 1])]
                continue  

        # --- YENİ KAYNAK HEDEFİ BELİRLEME, BEKLEME İSTASYONU VEYA MOLOZ TEMİZLİĞİ ---
        # 1. Bekleme İstasyonu: Eğar unit dolu degilse, işi bittiyse ve sarji yerindeyse.
        if total_cargo == 0 and unit_power > 1500 and home_pos is not None:
            # Fabrika etrafindaki rubble dolu Chebyshev==2 bloklarina git
            best_park_pos = None
            best_rubble = -1
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if max(abs(dx), abs(dy)) == 2:
                        rx, ry = home_pos[0] + dx, home_pos[1] + dy
                        if 0 <= rx < rubble.shape[0] and 0 <= ry < rubble.shape[1]:
                            if (rx, ry) not in GLOBAL_RESERVED_TILES:
                                # Daha cok rubble olan yere park et ki orayi temizlesin
                                r_val = rubble[rx, ry]
                                if r_val > best_rubble:
                                    best_rubble = r_val
                                    best_park_pos = np.array([rx, ry])
                                    
            if best_park_pos is not None:
                GLOBAL_RESERVED_TILES.add((best_park_pos[0], best_park_pos[1]))
                # Sadece moloz varsa parkta kal, yoksa diger kaynaklara dogru Ejection baslasin.
                if best_rubble > 0 or np.array_equal(pos, home_pos):
                    target_pos = best_park_pos
        
        # Eger hala target_pos atanmadiysa (veya rubble kalmadiysa), Ice/Ore ara
        if target_pos is None:
            current_target_pref = "ice" if home_water < 300 else my_role
            resource_coords = ice_coords if current_target_pref == "ice" else ore_coords
            
            if len(resource_coords) > 0:
                dist = np.abs(resource_coords - pos).sum(axis=1)
                sorted_idx = np.argsort(dist)
                target_idx = sorted_idx[0]
                if len(sorted_idx) > 1 and np.random.rand() < 0.10:
                    target_idx = sorted_idx[1]
                target_pos = resource_coords[target_idx]

        if target_pos is not None:
            if np.array_equal(pos, target_pos):
                actions[unit_id] = [np.array([3, 0, 0, 0, 0, 1])]   # DIG
            else:
                # Rastgele Keşif ve Rubble Temizleme
                cx, cy = pos[0], pos[1]
                curr_rubble = rubble[cx, cy] if cx < rubble.shape[0] and cy < rubble.shape[1] else 0
                
                # Eğer rubble varsa %15 ihtimalle, rubble yoksa %5 ihtimalle yolda "Dig" yaparak keşfe zaman ayır
                if (curr_rubble > 0 and np.random.rand() < 0.15) or np.random.rand() < 0.05:
                    actions[unit_id] = [np.array([3, 0, 0, 0, 0, 1])]   # DIG (Rubble temizliği / Mola)
                else:
                    direction = get_safe_direction(pos, target_pos, enemy_factory_tiles)
                    actions[unit_id] = [np.array([0, direction, 0, 0, 0, 1])]  # MOVE

    for f_id, f_data in factories.items():
        cargo = f_data.get("cargo", {})
        # Fabrikanın robot üretmesi için 2500 güç bariyeri (Güçlü Doğum)
        if cargo.get("metal", 0) >= 100 and f_data.get("power", 0) >= 2500:
            # Collision Prevention (Spawn on Occupied Center)
            center_x, center_y = f_data["pos"]
            occupying_unit = None
            for uid, udata in units.items():
                if udata["pos"][0] == center_x and udata["pos"][1] == center_y:
                    occupying_unit = uid
                    break
            
            if occupying_unit is not None:
                # EVICT! Assign a move command to the unit immediately to avoid being crushed.
                # Just randomly step out to an adjacent tile.
                force_direction = 1 if center_y > 0 else 3 # up
                actions[occupying_unit] = [np.array([0, force_direction, 0, 0, 0, 1])]
            
            actions[f_id] = 1   # Heavy Robot üret
        elif cargo.get("water", 0) >= 100 and f_data.get("power", 0) >= 50:
            actions[f_id] = 2   # Lichen sula (yalnızca su fazlaysa)

    return actions


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(">>> Initializing Project Sentinel Environment Context")
    # verbose=0 → terminal'i gereksiz log ile kirletme
    base_env = LuxAI_S2(verbose=0)
    env      = LuxS2Wrapper(base_env)
    logger   = TrajectoryLogger(output_dir="dataset_expert")
    agent    = OfflineCQLAgent(in_channels=env.num_channels, action_dim=64, map_size=env.map_size)

    print(">>> Running Full Episode Data Collection (300 Matches)...")
    total_steps_collected = 0
    NUM_EPISODES = 20

    for ep in range(76, 76 + NUM_EPISODES):
        # Her maça tertemiz bir hafızayla başla
        UNIT_HOME_MAP.clear()
        UNIT_ROLE_MAP.clear()
        UNIT_STUCK_TIME.clear()
        UNIT_PREV_POS.clear()
        GLOBAL_HUB_BUILT.clear()
        GLOBAL_HUB_BUILT.update({"player_0": False, "player_1": False})
        
        seed = np.random.randint(0, 100000)
        raw_obs, _ = base_env.reset(seed=seed)
        obs = env._process_obs(raw_obs)
        
        replay_steps = []
        step = 0
        dist_val = "N/A"

        while True:
            # Her tur başında "bu tur yapılacak spawnlar" listesini temizle
            GLOBAL_PENDING_SPAWNS = {"player_0": [], "player_1": []}
            GLOBAL_RESERVED_TILES.clear()
            
            p0_actions = get_heuristic_actions(raw_obs, "player_0", base_env)
            p1_actions = get_heuristic_actions(raw_obs, "player_1", base_env)
            combined   = {"player_0": p0_actions, "player_1": p1_actions}

            flat_state = raw_obs.get("player_0", raw_obs)
            obs_str    = json.dumps(flat_state, cls=NpEncoder)

            step_entry = []
            for player_idx, pact in enumerate([p0_actions, p1_actions]):
                step_entry.append({
                    "action": json.loads(json.dumps(pact, cls=NpEncoder)),
                    "info":   {},
                    "observation": {
                        "obs":                  obs_str,
                        "player":               player_idx,
                        "remainingOverageTime": 60,
                        "step":                 step,
                    },
                    "reward": 0.0,
                    "status": "ACTIVE",
                })
            replay_steps.append(step_entry)

            next_cnn_obs, rewards, global_terminated, global_truncated, _ = env.step(combined)

            flat_obs = base_env.state.get_obs()
            raw_obs  = {"player_0": flat_obs, "player_1": flat_obs}

            if dist_val == "N/A" and base_env.state.real_env_steps >= 0:
                facs0 = flat_obs.get("factories", {}).get("player_0", {})
                facs1 = flat_obs.get("factories", {}).get("player_1", {})
                if len(facs0) > 0 and len(facs1) > 0:
                    p0_f = list(facs0.values())[0]["pos"]
                    p1_f = list(facs1.values())[0]["pos"]
                    dist_val = int(np.abs(np.array(p0_f) - np.array(p1_f)).sum())

            logger.log_step(step, obs, p0_actions, rewards, global_terminated)

            # Reduce console spam for 75 matches
            if step > 0 and step % 500 == 0:
                print(f"  [Ep {ep} | Step {step:03d}] Processing...")

            obs = next_cnn_obs
            step += 1
            total_steps_collected += 1

            if global_terminated or global_truncated:
                flat_final = raw_obs.get("player_0", raw_obs)
                final_str  = json.dumps(flat_final, cls=NpEncoder)
                replay_steps.append([
                    {"action": {}, "info": {}, "observation": {"obs": final_str, "player": 0, "remainingOverageTime": 60, "step": step}, "reward": rewards, "status": "DONE"},
                    {"action": {}, "info": {}, "observation": {"obs": final_str, "player": 1, "remainingOverageTime": 60, "step": step}, "reward": 0.0,    "status": "DONE"},
                ])
                break

        # Calculate Post-Episode Summary
        team0_water = sum([f.get("cargo", {}).get("water", 0) for f in flat_obs.get("factories", {}).get("player_0", {}).values()])
        team1_water = sum([f.get("cargo", {}).get("water", 0) for f in flat_obs.get("factories", {}).get("player_1", {}).values()])
        
        status = "Draw"
        if team0_water > team1_water:
            status = "Player 0 Wins"
        elif team0_water < team1_water:
            status = "Player 1 Wins"
        print(f"\nEpisode [{ep}/375] (Seed: {seed}) | Steps: {step} | Status: {status} | Distance: {dist_val}")

        # Save parquet with desired naming
        logger.flush_episode(episode_id=f"sentinel_batch1_{ep}")

        # Keep s2vis for just the last episode to prevent IO bottleneck overhead
        if ep == 76 + NUM_EPISODES - 1:
            print(">>> Generating replay.json for the final episode (s2vis compatible)...")
            try:
                kaggle_replay = {
                    "steps": replay_steps,
                    "configuration": {
                        "actTimeout":    3,
                        "agentTimeout":  60,
                        "episodeSteps":  base_env.env_cfg.max_episode_length,
                        "mapSize":       base_env.env_cfg.map_size,
                        "maxFactories":  base_env.env_cfg.MAX_FACTORIES,
                        "seed":          int(seed),
                        "verbose":       0,
                    },
                    "info":     {"TeamNames": ["player_0", "player_1"]},
                    "rewards":  [0, 0],
                    "statuses": ["DONE", "DONE"],
                }
                with open("replay.json", "w") as f:
                    json.dump(kaggle_replay, f, cls=NpEncoder)
                print(f"[-] replay.json saved ({len(replay_steps)} steps) -> https://s2vis.lux-ai.org")
            except Exception as e:
                import traceback
                print(f"[!] Replay generation failed: {e}")
                traceback.print_exc()

    print(f"\n>>> Sentinel Data Collection Complete! Total Steps Built: {total_steps_collected}")


if __name__ == "__main__":
    main()
