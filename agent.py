import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from autodp import mechanism_zoo

class OfflineCQLAgent(nn.Module):
    """
    Conservative Q-Learning (CQL) Structure for Offline Reinforcement Learning,
    integrating Differential Privacy (DP) for preserving "Signature Moves".
    """
    def __init__(self, in_channels, action_dim, map_size=48, num_ensembles=3):
        super().__init__()
        self.action_dim = action_dim
        self.num_ensembles = num_ensembles
        
        # Spatial Feature Extractor Ensembles for Variance Estimation (LCB)
        self.cnn_ensembles = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * map_size * map_size, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            ) for _ in range(num_ensembles)
        ])
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
        # DP tracking budget
        self.total_epsilon = 0.0

    def forward(self, state):
        """
        Average Ensemble predictions for stable Q-values.
        """
        q_vals = torch.stack([cnn(state) for cnn in self.cnn_ensembles], dim=0)
        return torch.mean(q_vals, dim=0)

    def calculate_policy_entropy(self, state, temperature=1.0):
        """
        The "Signature Move" Detector.
        Calculates entropy over the ensemble's consensus Q-values.
        Low entropy = High expert certainty (i.e. 'Signature Move').
        """
        with torch.no_grad():
            q_vals = self.forward(state)
            probs = F.softmax(q_vals / temperature, dim=-1)
            # Add small eps to prevent log(0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy

    def compute_lcb_penalty(self, state, action_idx):
        """
        Variance-Aware Lower Confidence Bound (LCB).
        Computes standard deviation among ensemble Q-value predictions for a given action.
        Serves to penalize high-variance / Out-of-Distribution transitions!
        """
        q_vals_ensemble = torch.stack([cnn(state) for cnn in self.cnn_ensembles], dim=0)
        
        # Gather the specific action's Q-values from each ensemble branch
        action_idx = action_idx.unsqueeze(-1).long()
        action_qs = torch.stack([q.gather(1, action_idx) for q in q_vals_ensemble], dim=0)
        
        # Standard deviation calculates uncertainty divergence across models
        q_std = action_qs.std(dim=0).squeeze(-1) 
        return q_std

    def dp_sgd_step(self, obs_batch, action_batch, reward_batch, next_obs_batch, entropy_threshold=2.0):
        """
        Batched training logic bridging conservative offline calculations with 
        Differential Privacy constraints and utility budget tracing!
        """
        self.optimizer.zero_grad()
        
        # 1. Signature Move Detective
        entropies = self.calculate_policy_entropy(obs_batch)
        is_signature_move = torch.median(entropies).item() < entropy_threshold
        
        # 2. Offline CQL Batch Logic 
        with torch.no_grad():
            next_qs = self.forward(next_obs_batch).max(dim=-1)[0]
            target_q = reward_batch + 0.99 * next_qs # Standard Bellman Target
            
        current_qs = self.forward(obs_batch).gather(1, action_batch.unsqueeze(-1).long()).squeeze(-1)
        
        # Baseline Temporal Difference Loss
        mse_loss = F.mse_loss(current_qs, target_q, reduction='none')
        
        # Subtract Pessimism Penalty (LCB) on OOD expert actions
        lcb_penalty = self.compute_lcb_penalty(obs_batch, action_batch)
        
        # Minimize (MSE + LCB) to constrain overestimation
        loss = (mse_loss + 0.1 * lcb_penalty).mean()
        loss.backward()
        
        # 3. Adaptive DP & Automatic Clipping
        clip_max_norm = 1.0 
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_max_norm)
        
        if is_signature_move:
            print(f"   [Privacy Shield] Signature Move detected (Entropy < {entropy_threshold}). Obscuring gradient!")
            
            # Formally calculate mechanism noise using autodp scaling
            sigma = 0.5
            noise_scale = sigma * clip_max_norm 
            
            for param in self.parameters():
                if param.grad is not None:
                    noise = torch.normal(mean=0.0, std=noise_scale, size=param.grad.shape).to(param.device)
                    param.grad += noise
                    
            # 4. Consume Bugdet via Autodp
            mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
            eps_spent = mech.get_approxDP(delta=1e-5)
            self.total_epsilon += eps_spent
            print(f"   -> DP Epsilon Expenditure: {eps_spent:.4f} | Total Epsilon Budget Consumed: {self.total_epsilon:.4f}")
            
        self.optimizer.step()
        return loss.item(), is_signature_move
