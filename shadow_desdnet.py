import torch
import torch.nn as nn
import torchvision.models as models

class DynamicAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))

class ShadowDESDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ResNeXt50
        backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        
        # Extract feature layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Attention module
        self.attention = DynamicAttention(2048)
        
        # Output layers
        self.output = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        
        # Apply attention
        attended = self.attention(features)
        
        # Generate output
        return self.output(attended)