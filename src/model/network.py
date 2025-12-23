"""
Mario RL Network combining backbone (MobileNetV2 or ConvNeXt V2) with actor-critic heads.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .backbone import get_backbone


class MarioNetwork(nn.Module):
    """
    Actor-Critic network for Mario RL agent.
    
    Architecture:
        - Backbone (MobileNetV2 or ConvNeXt V2) for feature extraction
        - Actor head: MLP that outputs action logits
        - Critic head: MLP that outputs state value
    
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: 
        - action_logits: (batch, num_actions)
        - state_value: (batch, 1)
    """
    
    def __init__(
        self,
        num_actions: int = 4,
        hidden_units: int = 512,
        backbone_type: str = "mobilenetv2",
        backbone_pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the Mario network.
        
        Args:
            num_actions: Number of discrete actions (default: 4)
            hidden_units: Size of hidden layer in MLP heads (default: 512)
            backbone_type: Type of backbone ("mobilenetv2", "convnextv2")
            backbone_pretrained: Whether to use pretrained backbone
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Create backbone using factory function
        self.backbone = get_backbone(
            backbone_type=backbone_type,
            pretrained=backbone_pretrained,
            freeze=freeze_backbone
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        
        # Initialize actor and critic weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for actor and critic heads."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84)
            
        Returns:
            Tuple of:
                - action_logits: (batch, num_actions)
                - state_value: (batch, 1)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Get action logits from actor head
        action_logits = self.actor(features)
        
        # Get state value from critic head
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84)
            deterministic: If True, return argmax action; else sample
            
        Returns:
            Tuple of:
                - action: (batch,) selected action indices
                - log_prob: (batch,) log probability of selected actions
                - value: (batch, 1) state value estimate
        """
        action_logits, value = self.forward(x)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84)
            actions: Action indices of shape (batch,)
            
        Returns:
            Tuple of:
                - log_prob: (batch,) log probability of actions
                - value: (batch, 1) state value estimate
                - entropy: (batch,) entropy of action distribution
        """
        action_logits, value = self.forward(x)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


if __name__ == "__main__":
    # Test the network
    net = MarioNetwork(num_actions=4, hidden_units=512)
    x = torch.randn(2, 4, 84, 84)
    
    # Test forward pass
    logits, value = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Action logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test get_action
    action, log_prob, value = net.get_action(x)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    # Test evaluate_actions
    log_prob, value, entropy = net.evaluate_actions(x, action)
    print(f"Entropy shape: {entropy.shape}")
