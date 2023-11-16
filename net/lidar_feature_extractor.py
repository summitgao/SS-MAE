import torch
import torch.nn as nn

class lidar_e(nn.Module):
    def __init__(self,args):
        if args.dataset == "Houston2018":
            in_chan = 1
        else:
            in_chan = 4
        super(lidar_e, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=in_chan, out_channels=64, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
            )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64))
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x