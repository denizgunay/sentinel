import torch
from luxai_s2.env import LuxAI_S2
from env_wrapper import LuxS2Wrapper
from logger import TrajectoryLogger
from agent import OfflineCQLAgent

def main():
    print(">>> Initializing Project Sentinel Environment Context")
    
    # Environment Setup
    base_env = LuxAI_S2()
    env = LuxS2Wrapper(base_env)
    
    # Utilities & Storage
    logger = TrajectoryLogger(output_dir="dataset_expert")
    
    # Init Batched CQL Agent instance
    agent = OfflineCQLAgent(in_channels=env.num_channels, action_dim=64, map_size=env.map_size) 
    
    obs, info = env.reset()
    
    print(">>> Mocking Batched Data Processing Loop...")
    for step in range(5): # Very short loop to demonstrate the agent logs
        mock_action = {"player_0": {}, "player_1": {}}
        
        # Environment physics step
        next_obs, rewards, global_terminated, global_truncated, infos = env.step(mock_action)
        
        # Logging & tracking for our Offline Dataset
        logger.log_step(step, obs, mock_action, rewards, global_terminated)
        
        # --- Batching Transitions for the Custom SGD Agent ---
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        
        # The NN actions would normally be mapped outputs. 
        # Simulating selecting index action 5!
        action_tensor = torch.tensor([5], dtype=torch.long)
        reward_tensor = torch.tensor([rewards], dtype=torch.float32)
        
        # Step the Batched CQL Agent with Auto Differential Privacy Check
        print(f"\\n--- Frame Step {step} ---")
        loss, was_signature = agent.dp_sgd_step(
            state_tensor, 
            action_tensor, 
            reward_tensor, 
            next_state_tensor, 
            entropy_threshold=2.0 # Threshold calibrated artificially high to guarantee trigger for demo
        )
        print(f"   -> MSE + LCB Loss: {loss:.4f}")

        obs = next_obs
        if global_terminated or global_truncated:
            break
            
    # Flush batch strictly at EOF (user requirement 1)
    print("\\n>>> Loop Finished. Exporting Episode Dataset...")
    logger.flush_episode(episode_id="dev_test_001")
    print(">>> Sentinel Runtime Architecture Complete!")

if __name__ == "__main__":
    main()
