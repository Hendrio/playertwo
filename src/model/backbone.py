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


class ConvNeXtV2Backbone(nn.Module):
    """
    Modified ConvNeXt V2 backbone for Mario RL agent.
    
    ConvNeXt V2 offers improved feature extraction compared to MobileNetV2.
    Uses the Tiny variant for balance of speed and quality.
    
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 768) - feature vector
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False, variant: str = "tiny"):
        """
        Initialize ConvNeXt V2 backbone.
        
        Args:
            pretrained: Whether to load ImageNet pretrained weights
            freeze: Whether to freeze backbone weights
            variant: Model variant ("tiny", "small", "base")
        """
        super().__init__()
        
        from torchvision.models import convnext_tiny, convnext_small, convnext_base
        from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
        
        # Select variant
        variant_configs = {
            "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
            "small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
            "base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        }
        
        if variant not in variant_configs:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(variant_configs.keys())}")
        
        model_fn, weights_class, self._output_dim = variant_configs[variant]
        
        # Load pretrained model
        if pretrained:
            base_model = model_fn(weights=weights_class)
        else:
            base_model = model_fn(weights=None)
        
        # Get the feature extractor (everything except classifier)
        self.features = base_model.features
        
        # Modify first convolutional layer for 4-channel input
        # ConvNeXt first layer: Conv2d(3, 96, kernel_size=4, stride=4) for tiny
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
        return self._output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (batch, 4, 84, 84)
            
        Returns:
            Feature tensor of shape (batch, 768) or (batch, 1024) for base
        """
        # Features extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.pool(x)
        
        # Flatten to (batch, output_dim)
        x = torch.flatten(x, 1)
        
        return x


def get_backbone(backbone_type: str, pretrained: bool = True, freeze: bool = False) -> nn.Module:
    """
    Factory function to get backbone by name.
    
    Args:
        backbone_type: Type of backbone ("mobilenetv2", "convnextv2")
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone weights
        
    Returns:
        Backbone module
    """
    backbone_type = backbone_type.lower()
    
    if backbone_type == "mobilenetv2":
        return MobileNetV2Backbone(pretrained=pretrained, freeze=freeze)
    elif backbone_type in ("convnextv2", "convnext"):
        return ConvNeXtV2Backbone(pretrained=pretrained, freeze=freeze)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Choose from: mobilenetv2, convnextv2")


if __name__ == "__main__":
    # Test both backbones
    print("Testing MobileNetV2Backbone:")
    model1 = MobileNetV2Backbone(pretrained=True, freeze=False)
    x = torch.randn(2, 4, 84, 84)
    out1 = model1(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out1.shape}")
    print(f"  Output dim: {model1.output_dim}")
    
    print("\nTesting ConvNeXtV2Backbone:")
    model2 = ConvNeXtV2Backbone(pretrained=True, freeze=False)
    out2 = model2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out2.shape}")
    print(f"  Output dim: {model2.output_dim}")
    
    print("\nTesting get_backbone factory:")
    for name in ["mobilenetv2", "convnextv2"]:
        backbone = get_backbone(name)
        print(f"  {name}: output_dim={backbone.output_dim}")
