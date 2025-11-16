import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolution + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class EnhancedBrainTumorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EnhancedBrainTumorCNN, self).__init__()

        self.layer1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2)  # 224 → 112
        )

        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64, dropout=0.1),
            nn.MaxPool2d(2, 2)  # 112 → 56
        )

        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128, dropout=0.2),
            nn.MaxPool2d(2, 2)  # 56 → 28
        )

        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256, dropout=0.3),
            nn.MaxPool2d(2, 2)  # 28 → 14
        )

        self.layer5 = nn.Sequential(
            ConvBlock(256, 512, dropout=0.4),
            ConvBlock(512, 512, dropout=0.4),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        out = self.fc(x5)
        return out
