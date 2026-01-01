import torch
import torch.nn as nn
import torch.nn.functional as F

class AMLSCNN(nn.Module):
    def __init__(self, num_classes=2, channels=[8, 16, 32], dropout=0.3):
        """Simple CNN for 28x28 grayscale images with dropout for regularization."""
        super().__init__()

        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)  # 添加批归一化
        
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)  # 添加dropout

        # dummy forward pass to compute fc input dim
        dummy = torch.zeros(1, 1, 28, 28)
        out = self._forward_features(dummy)
        flat_dim = out.view(1, -1).size(1)

        self.fc = nn.Linear(flat_dim, num_classes)


    def _forward_features(self, x):
        """Feature extraction layers with batch normalization."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28->14
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14->7
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))             # 7->7
        return x

    def forward(self, x):
        """Forward pass."""
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
