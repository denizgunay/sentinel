import pandas as pd
import uuid
import os

class TrajectoryLogger:
    """
    Logs trajectories (obs, actions, rewards) to Pandas Dataframes
    and flushes to Parquet per episode for robust Offline RL storage.
    """
    def __init__(self, output_dir="dataset_expert"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.trajectory_buffer = []
        
        # Pre-check pyarrow as requested by user
        try:
            import pyarrow
            self.has_pyarrow = True
            print("[-] PyArrow detected. Parquet logging: ENABLED.")
        except ImportError:
            self.has_pyarrow = False
            print("[!] WARNING: PyArrow not found. Falling back to CSV storage.")
            
    def log_step(self, step, obs, action, reward, done):
        """
        Records the components of a transition into the memory list.
        """
        # Ensure we capture a serializable version of the CNN observation
        # For Parquet/Arrow, we keep it as a flat list for simple storage
        obs_data = obs.flatten().tolist() if hasattr(obs, 'flatten') else obs
        
        self.trajectory_buffer.append({
            "step": step,
            "observation": obs_data,
            "action": str(action), # Safely serializing action dictionaries/lists
            "reward": float(reward),
            "done": bool(done)
        })
        
    def flush_episode(self, episode_id=None):
        """
        Flushes the current episode trajectory buffer to a single Parquet file.
        """
        if not self.trajectory_buffer:
            return
            
        if episode_id is None:
            episode_id = str(uuid.uuid4())[:8]
            
        df = pd.DataFrame(self.trajectory_buffer)
        
        if self.has_pyarrow:
            file_path = os.path.join(self.output_dir, f"episode_{episode_id}.parquet")
            try:
                df.to_parquet(file_path, engine='pyarrow', index=False)
                print(f"[-] Successfully saved episode DB to PARQUET: {file_path}")
            except Exception as e:
                print(f"[!] Parquet save failed: {e}. Falling back to CSV.")
                self._save_csv(df, episode_id)
        else:
            self._save_csv(df, episode_id)
            
        # Clear buffer
        self.trajectory_buffer = []

    def _save_csv(self, df, episode_id):
        file_path = os.path.join(self.output_dir, f"episode_{episode_id}.csv")
        df.to_csv(file_path, index=False)
        print(f"[-] Successfully saved episode DB to CSV: {file_path}")
