from torch import nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=1 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2),
        #stride = 3 (x3 scaling), kernel = 5 => padding = 1
        self.conv3 = nn.ConvTranspose2d(32, 3, 5, 3, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
from torch import nn


class SRCNNResidual(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNNResidual, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=1 // 2)
        self.conv3 = nn.ConvTranspose2d(32, num_channels, kernel_size=5, stride=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=3, mode="bicubic", align_corners=False)  # 입력을 x3 업샘플링
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x += residual  # 잔차 더하기
        return x
