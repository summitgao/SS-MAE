import torch.nn as nn

class hsi_e(nn.Module):
    def __init__(self, args):
        super(hsi_e, self).__init__()
        self.args = args
        self.conv2d1_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.pca_num, out_channels=args.pca_num, kernel_size=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(num_features=args.pca_num)
        )
        self.conv3d1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(9, 3, 3), padding=1),
            nn.GELU(),
            nn.BatchNorm3d(num_features=4))
        self.conv2d1_2 = nn.Sequential(
            nn.Conv2d(in_channels=(self.args.pca_num - 6) * 4, out_channels=(self.args.pca_num - 6) * 8,
                      kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=(self.args.pca_num - 6) * 8)
        )

        self.conv2d2_1 = nn.Sequential(
            nn.Conv2d(in_channels=(self.args.pca_num - 6) * 8, out_channels=(self.args.pca_num - 6) * 4,
                      kernel_size=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(num_features=(self.args.pca_num - 6) * 4)
        )
        self.conv3d2 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(7, 3, 3), padding=1),
            nn.GELU(),
            nn.BatchNorm3d(num_features=8))
        self.conv2d2_2 = nn.Sequential(
            nn.Conv2d(in_channels=(self.args.pca_num - 10) * 8, out_channels=(self.args.pca_num - 10) * 16,
                      kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=(self.args.pca_num - 10) * 16)
        )

        self.conv2d3_1 = nn.Sequential(
            nn.Conv2d(in_channels=(self.args.pca_num - 10) * 16, out_channels=(self.args.pca_num - 10) * 8,
                      kernel_size=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(num_features=(self.args.pca_num - 10) * 8)
        )
        self.conv3d3 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=1),
            nn.GELU(),
            nn.BatchNorm3d(num_features=16))
        self.conv2d3_2 = nn.Sequential(
            nn.Conv2d(in_channels=(self.args.pca_num - 12) * 16, out_channels=(self.args.pca_num - 12) * 32,
                      kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=(self.args.pca_num - 12) * 32)
        )
        self.hsi_conv = nn.Sequential(
            nn.Conv2d(in_channels=32 * (self.args.pca_num - 12), out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv2d1_1(x)
        x = self.conv3d1(x.unsqueeze(1))
        x = self.conv2d1_2(x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]))
        x = self.conv2d2_1(x)
        x = self.conv3d2(x.view(x.shape[0], 4, -1, x.shape[-2], x.shape[-1]))
        x = self.conv2d2_2(x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]))
        x = self.conv2d3_1(x)
        x = self.conv3d3(x.view(x.shape[0], 8, -1, x.shape[-2], x.shape[-1]))
        x = self.conv2d3_2(x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]))
        x = self.hsi_conv(x)
        return x
        

