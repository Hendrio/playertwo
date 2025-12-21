"""
MobileNetV2 backbone modified for 4-channel grayscale frame input.

The standard MobileNetV2 expects RGB (3-channel) input. This module modifies
the first convolutional layer to accept stacked grayscale frames (4 channels).
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Backbone(nn.Module):
    """
    Modified MobileNetV2 backbone for Mario RL agent.
    
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 1280) - feature vector
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        """
        Initialize MobileNetV2 backbone.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            base_model = mobilenet_v2(weights=weights)
        else:
            base_model = mobilenet_v2(weights=None)
        
        # Get the feature extractor (everything except classifier)
        self.features = base_model.features
        
        # Modify first convolutional layer for 4-channel input
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            in_channels=4,  # Changed from 3 to 4 for stacked frames
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize new conv layer weights
        if pretrained:
            # Average the RGB weights and replicate for 4 channels
            with torch.no_grad():
                rgb_weights = original_conv.weight.data
                # Average across RGB channels: (out, 3, H, W) -> (out, 1, H, W)
                avg_weights = rgb_weights.mean(dim=1, keepdim=True)
                # Replicate for 4 channels: (out, 1, H, W) -> (out, 4, H, W)
                self.features[0][0].weight.data = avg_weights.repeat(1, 4, 1, 1)
        
        # Global average pooling to get fixed-size output
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone if specified
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return 1280
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84)
            
        Returns:
            Feature tensor of shape (batch, 1280)
        """
        # Features extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.pool(x)
        
        # Flatten to (batch, 1280)
        x = torch.flatten(x, 1)
        
        return x


if __name__ == "__main__":
    # Test the backbone
    model = MobileNetV2Backbone(pretrained=True, freeze=False)
    x = torch.randn(2, 4, 84, 84)  # Batch of 2, 4 stacked frames, 84x84
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dim: {model.output_dim}")
