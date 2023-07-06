import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_dims, out_dims,  mid_dims=False):
        super(DoubleConv, self).__init__()

        if not mid_dims:
            mid_dims = out_dims

        self.conv_1 = nn.Conv2d(in_dims, mid_dims, kernel_size=(3, 3), padding=1, bias=False)
        self.norm_1 = nn.BatchNorm2d(mid_dims)
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(mid_dims, out_dims, kernel_size=(3, 3), padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_dims)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.relu_2(x)

        return x


class DownDoubleConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DownDoubleConv, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_dims, out_dims)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


class UpDoubleConv(nn.Module):
    def __init__(self, in_dims, out_dims, bilinear=True):
        super(UpDoubleConv, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_dims, out_dims, in_dims//2)

        else:
            self.up = nn.ConvTranspose2d(in_dims, in_dims//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_dims, out_dims)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FullScaleConv(nn.Module):
    def __init__(self, factor, in_dims, out_dims, sampling=None):
        super(FullScaleConv, self).__init__()

        self.sampling = sampling

        if self.sampling == 'up':
            self.up = nn.Upsample(scale_factor=factor, mode='bilinear')
        elif self.sampling == 'down':
            self.pool = nn.MaxPool2d(factor, factor, ceil_mode=True)

        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.sampling == 'up':
            x = self.up(x)
        elif self.sampling == 'down':
            x = self.pool(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FusionConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(FusionConv, self).__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x