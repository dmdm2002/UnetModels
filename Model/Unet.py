import torch
import torch.nn as nn

from torchsummary.torchsummary import summary
from Model.UnetParts import DoubleConv, DownDoubleConv, UpDoubleConv


class Unet_model(nn.Module):
    def __init__(self, n_dims, n_classes, bilinear=True):
        super(Unet_model, self).__init__()
        self.n_dims = n_dims
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        # 224 x 224 x 64
        self.double_conv = DoubleConv(self.n_dims, 64)

        # 224 x 224 x 64
        self.pool = nn.MaxPool2d(2)
        self.double_1 = DoubleConv(64, 128)
        self.down_double_conv_1 = DownDoubleConv(64, 128)
        # 112 x 112 x 64 --> 112 x 112 x 128
        self.down_double_conv_2 = DownDoubleConv(128, 256)
        # 56 x 56 x 256 --> 56 x 56 x 512
        self.down_double_conv_3 = DownDoubleConv(256, 512)

        self.down_double_conv_4 = DownDoubleConv(512, 1024//factor)

        self.up_double_conv_1 = UpDoubleConv(1024, 512//factor, bilinear)
        self.up_double_conv_2 = UpDoubleConv(512, 256//factor, bilinear)
        self.up_double_conv_3 = UpDoubleConv(256, 128//factor, bilinear)
        self.up_double_conv_4 = UpDoubleConv(128, 64, bilinear)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv(x)

        x2 = self.pool(x1)
        x2 = self.double_1(x2)
        # x2 = self.down_double_conv_1(x1)
        x3 = self.down_double_conv_2(x2)
        x4 = self.down_double_conv_3(x3)
        x5 = self.down_double_conv_4(x4)

        x_up = self.up_double_conv_1(x5, x4)
        x_up = self.up_double_conv_2(x_up, x3)
        x_up = self.up_double_conv_3(x_up, x2)
        x_up = self.up_double_conv_4(x_up, x1)

        output = self.out(x_up)

        return output


model = Unet_model(3, 1)
summary(model, (3, 224, 224), device='cpu')
