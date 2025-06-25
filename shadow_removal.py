import torch
import torch.nn as nn

class ShadowRemoval(nn.Module):
    def __init__(self):
        super().__init__()
        # More memory-efficient architecture
        self.net = nn.Sequential(
            # Downsample first to reduce memory
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Middle layers
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Upsample back
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, image, shadow_map):
        # Ensure matching sizes
        if shadow_map.size()[2:] != image.size()[2:]:
            shadow_map = nn.functional.interpolate(shadow_map, size=image.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([image, shadow_map], dim=1)
        return self.net(x)