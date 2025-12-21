"""
Proximal Policy Optimization (PPO) implementation.

PPO is a policy gradient method that uses a clipped surrogate objective
to ensure stable policy updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .schedules import LinearSchedule


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout experiences."""
    
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    values: List[float]
    log_probs: List[float]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def __len__(self):
        return len(self.observations)


class PPOAgent:
    """
    PPO Agent for training Mario.
    
    Uses Generalized Advantage Estimation (GAE) for advantage computation
    and clipped surrogate objective for policy updates.
    """
    
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 32,
        total_steps: int = 10000000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            network: Actor-critic network
            learning_rate: Initial learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of optimization epochs per rollout
            batch_size: Mini-batch size
            total_steps: Total training steps (for LR scheduling)
            device: Device to use (cuda/cpu)
        """
        self.network = network.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        
        # Learning rate schedule
        self.lr_schedule = LinearSchedule(learning_rate, 0.0, total_steps)
        
        # Epsilon (exploration) schedule
        self.epsilon_schedule = LinearSchedule(1.0, 0.1, total_steps)
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Training step counter
        self.global_step = 0
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array (4, 84, 84)
            deterministic: If True, select best action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)
        
        # Epsilon-greedy exploration
        epsilon = self.epsilon_schedule(self.global_step)
        if not deterministic and np.random.random() < epsilon:
            action = torch.tensor([np.random.randint(0, 4)], device=self.device)
            # Recalculate log_prob for random action
            logits, _ = self.network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
        
        return (
            action.item(),
            log_prob.item(),
            value.item()
        )
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Store a transition in the buffer."""
        self.buffer.add(obs, action, reward, done, value, log_prob)
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            last_value: Value estimate for the last state
            
        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        
        # Compute GAE
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """
        Perform PPO update using collected experiences.
        
        Args:
            last_obs: Last observation for bootstrapping value
            
        Returns:
            Dictionary of training metrics
        """
        # Get last value for GAE computation
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_obs_tensor)
            last_value = last_value.item()
        
        # Convert buffer to arrays
        observations = np.array(self.buffer.observations, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.int64)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)
        old_log_probs = np.array(self.buffer.log_probs, dtype=np.float32)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # Multiple epochs of optimization
        n_samples = len(observations)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    batch_obs, batch_actions
                )
                new_values = new_values.squeeze(-1)
                
                # Compute policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Update learning rate
        new_lr = self.lr_schedule(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update global step
        self.global_step += len(self.buffer)
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'learning_rate': new_lr,
            'epsilon': self.epsilon_schedule(self.global_step)
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from model.network import MarioNetwork
    
    # Create network and agent
    network = MarioNetwork(num_actions=4)
    agent = PPOAgent(network, learning_rate=1e-4)
    
    print(f"PPO Agent created")
    print(f"  Device: {agent.device}")
    print(f"  Global step: {agent.global_step}")
    
    # Test action selection
    obs = np.random.randn(4, 84, 84).astype(np.float32)
    action, log_prob, value = agent.select_action(obs)
    print(f"  Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
