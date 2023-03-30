import torch.nn as nn
import torch.nn.functional as F

# jo =jin*s
# rfo = rin+(k-1)*jin 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 1), # 32x32 - jin=1, jout=1, rf = 1+(4)*1 = 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, 2, padding=1), #strided, 32x32 - jin=1, jout=2, rf = 5+(2)*1 = 7
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1,dilation=1), # dilation, 16x16 - jo=2 - rf=7+(5-1)*2=15
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32), # depthwise-seperable, 16x16 - jo=2 - rf=15+(3-1)*2=19
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, padding=1), # strided, 16x16 - jo=4 - rf=19+(3-1)*2=23
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, padding=1), # 8x8 - jo=4 - rf=23+(3-1)*4=31
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv4 =  nn.Sequential(
            nn.Conv2d(128, 32, 3, 2, padding=1), # strided, 8x8 - jo=8 - rf=31+(3-1)*4=39
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1), # 4x4, Mix and Merge with 1x1
            nn.Conv2d(32, 32, 3, 1, 1), # 4x4 - jo=8 - rf=39+(3-1)*8=55
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
