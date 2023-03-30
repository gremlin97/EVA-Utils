import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1,dilation=1), # dilation
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32), # depthwise-seperable
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, padding=1), # strided
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv4 =  nn.Sequential(
            nn.Conv2d(128, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1)) #gap
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1,32)
        x = self.fc1(x)
        return x
