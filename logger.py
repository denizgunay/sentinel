import pandas as pd
import uuid
import os

class TrajectoryLogger:
    """
    Logs trajectories (obs, actions, rewards) to Pandas Dataframes
    and flushes to Parquet per episode for robust Offline RL storage.
    """
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.trajectory_buffer = []
        
    def log_step(self, step, obs, action, reward, done):
        """
        Records the components of a transition into the memory list.
        """
        self.trajectory_buffer.append({
            "step": step,
            # For 3D Tensors, we flatten or dump string repr for Parquet saving
            "observation": obs.flatten().tolist() if hasattr(obs, 'flatten') else obs,
            "action": str(action), # Safely serializing action dictionaries
            "reward": reward,
            "done": done
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
        file_path = os.path.join(self.output_dir, f"episode_{episode_id}.parquet")
        
        try:
            # PyArrow engine handles nested arrays nicely
            df.to_parquet(file_path, engine='pyarrow')
            print(f"[-] Successfully saved episode DB to {file_path}")
        except ImportError:
            print("[!] pyarrow not installed, falling back to CSV storage.")
            df.to_csv(file_path.replace(".parquet", ".csv"), index=False)
            
        # Clear buffer to free up memory constraints
        self.trajectory_buffer = []
