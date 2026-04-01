import gymnasium as gym
import numpy as np

class LuxS2Wrapper(gym.Env):
    """
    Gymnasium wrapper for Lux AI Season 2 focusing on Convolutional Neural Networks (CNN).
    Transforms complex dictionaries into a multi-channel 3D Tensor C x H x W.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Typically H, W for LuxS2 is defined in env_cfg
        self.map_size = env.env_cfg.map_size
        
        # Extracted Features (Channels):
        # 0: Ice Locations
        # 1: Ore Locations
        # 2: Own Factories
        # 3: Opponent Factories
        # 4: Own Robots
        # 5: Opponent Robots
        # 6: Power Map/Levels
        self.num_channels = 7
        
        # Overriding observation space to be CNN compatible
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_channels, self.map_size, self.map_size), 
            dtype=np.float32
        )
        
        # Explicit dummy action space to satisfy Gymnasium specifications
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(64,), 
            dtype=np.float32
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, rewards, terminations, truncations, infos = self.env.step(action)
        
        # Merge PettingZoo dictionary returns to Gymnasium singular variables
        global_terminated = any(terminations.values()) if isinstance(terminations, dict) else terminations
        global_truncated = any(truncations.values()) if isinstance(truncations, dict) else truncations
        global_reward = rewards.get("player_0", 0.0) if isinstance(rewards, dict) else rewards
        
        return self._process_obs(obs), global_reward, global_terminated, global_truncated, infos

    def _process_obs(self, obs):
        """
        Parses the raw nested dictionaries of Lux AI S2 into a 
        C x H x W float32 tensor representing spatial features.
        """
        H, W = self.map_size, self.map_size
        cnn_obs = np.zeros((self.num_channels, H, W), dtype=np.float32)
        
        # Process from player_0's perspective for offline logging boilerplating
        p0_obs = obs.get("player_0")
        if not p0_obs:
            return cnn_obs # Fallback for empty observation at end cases
            
        board = p0_obs.get("board", {})
        
        # Channel 0 & 1: Resources (Ice & Ore)
        if "ice" in board:
            cnn_obs[0] = board["ice"]
        if "ore" in board:
            cnn_obs[1] = board["ore"]
            
        # Channel 2-6: Factories, Units, and Power
        factories = p0_obs.get("factories", {})
        for player, pf in factories.items():
            for f_id, factory in pf.items():
                x, y = factory["pos"]
                channel = 2 if player == "player_0" else 3
                cnn_obs[channel, x, y] += 1
                
        units = p0_obs.get("units", {})
        for player, pu in units.items():
            for u_id, unit in pu.items():
                x, y = unit["pos"]
                channel = 4 if player == "player_0" else 5
                cnn_obs[channel, x, y] += 1
                cnn_obs[6, x, y] += unit["power"] # Power Map overlay
                
        return cnn_obs
