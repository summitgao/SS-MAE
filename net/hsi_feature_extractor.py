import torch.nn as nn

class hsi_e(nn.Module):
    def __init__(self,args):
        super(hsi_e, self).__init__()
        self.args = args

        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9, 3, 3), padding=1), #houston2018
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=8))

        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=16)
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=32)
        )

        self.hsi_conv = nn.Sequential(
            nn.Conv2d(in_channels=32 * (self.args.pca_num-12), out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
        )

    def forward(self, x):
        x = self.hsi_step1(x)
        x = self.hsi_step2(x)
        x = self.hsi_step3(x)
        x = x.reshape(-1, 32 * (self.args.pca_num-12), x.shape[3], x.shape[4])
        x = self.hsi_conv(x)
        return x

