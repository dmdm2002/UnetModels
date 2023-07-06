import torch
import torch.nn as nn

from torchsummary.torchsummary import summary
from Model.UnetParts import DoubleConv, FullScaleConv, FusionConv


class UNet3Plus(nn.Module):
    def __init__(self, in_dims=3, n_classes=1):
        super(UNet3Plus, self).__init__()
        self.in_dims = in_dims
        self.n_classes = n_classes

        self.n_filters = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.en_conv1 = DoubleConv(self.in_dims, self.n_filters[0])
        self.en_conv2 = DoubleConv(self.n_filters[0], self.n_filters[1])
        self.en_conv3 = DoubleConv(self.n_filters[1], self.n_filters[2])
        self.en_conv4 = DoubleConv(self.n_filters[2], self.n_filters[3])
        self.en_conv5 = DoubleConv(self.n_filters[3], self.n_filters[4])

        # Decoder
        self.cat_dims = self.n_filters[0]
        self.after_cat_dims = self.cat_dims * 5
        # Stage 4
        self.de_conv_4_1 = FullScaleConv(8, self.n_filters[0], self.cat_dims, sampling='down')
        self.de_conv_4_2 = FullScaleConv(4, self.n_filters[1], self.cat_dims, sampling='down')
        self.de_conv_4_3 = FullScaleConv(2, self.n_filters[2], self.cat_dims, sampling='down')
        self.de_conv_4_4 = FullScaleConv(1, self.n_filters[3], self.cat_dims)
        self.de_conv_4_5 = FullScaleConv(2, self.n_filters[4], self.cat_dims, sampling='up')

        self.de_conv4_fusion = FusionConv(self.after_cat_dims, self.after_cat_dims)

        # Stage 3
        self.de_conv_3_1 = FullScaleConv(4, self.n_filters[0], self.cat_dims, sampling='down')
        self.de_conv_3_2 = FullScaleConv(2, self.n_filters[1], self.cat_dims, sampling='down')
        self.de_conv_3_3 = FullScaleConv(1, self.n_filters[2], self.cat_dims)
        self.de_conv_3_4 = FullScaleConv(2, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_3_5 = FullScaleConv(4, self.n_filters[4], self.cat_dims, sampling='up')

        self.de_conv3_fusion = FusionConv(self.after_cat_dims, self.after_cat_dims)

        # Stage 2
        self.de_conv_2_1 = FullScaleConv(2, self.n_filters[0], self.cat_dims, sampling='down')
        self.de_conv_2_2 = FullScaleConv(1, self.n_filters[1], self.cat_dims)
        self.de_conv_2_3 = FullScaleConv(2, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_2_4 = FullScaleConv(4, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_2_5 = FullScaleConv(8, self.n_filters[4], self.cat_dims, sampling='up')

        self.de_conv2_fusion = FusionConv(self.after_cat_dims, self.after_cat_dims)

        # Stage 1
        self.de_conv_1_1 = FullScaleConv(1, self.n_filters[0], self.cat_dims)
        self.de_conv_1_2 = FullScaleConv(2, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_1_3 = FullScaleConv(4, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_1_4 = FullScaleConv(8, self.after_cat_dims, self.cat_dims, sampling='up')
        self.de_conv_1_5 = FullScaleConv(16, self.n_filters[4], self.cat_dims, sampling='up')

        self.de_conv1_fusion = FusionConv(self.after_cat_dims, self.after_cat_dims)

        self.outconv = nn.Conv2d(self.after_cat_dims, self.n_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.en_conv1(x) # 112

        x2 = self.pool(x1)
        x2 = self.en_conv2(x2) # 56

        x3 = self.pool(x2)
        x3 = self.en_conv3(x3) # 28

        x4 = self.pool(x3)
        x4 = self.en_conv4(x4) # 24

        x5 = self.pool(x4)
        x5_d = self.en_conv5(x5)

        # Decoder
        # Decoding x_4
        de_x_4_1 = self.de_conv_4_1(x1)
        de_x_4_2 = self.de_conv_4_2(x2)
        de_x_4_3 = self.de_conv_4_3(x3)
        de_x_4_4 = self.de_conv_4_4(x4)
        de_x_4_5 = self.de_conv_4_5(x5_d)

        de_x_4_cat = torch.cat([de_x_4_1, de_x_4_2, de_x_4_3, de_x_4_4, de_x_4_5], 1)
        de_x_4_d = self.de_conv4_fusion(de_x_4_cat)

        # Decoding x_3
        de_x_3_1 = self.de_conv_3_1(x1)
        de_x_3_2 = self.de_conv_3_2(x2)
        de_x_3_3 = self.de_conv_3_3(x3)
        de_x_3_4 = self.de_conv_3_4(de_x_4_d)
        de_x_3_5 = self.de_conv_3_5(x5_d)

        de_x_3_cat = torch.cat([de_x_3_1, de_x_3_2, de_x_3_3, de_x_3_4, de_x_3_5], 1)
        de_x_3_d = self.de_conv3_fusion(de_x_3_cat)

        # Decoding x_2
        de_x_2_1 = self.de_conv_2_1(x1)
        de_x_2_2 = self.de_conv_2_2(x2)
        de_x_2_3 = self.de_conv_2_3(de_x_3_d)
        de_x_2_4 = self.de_conv_2_4(de_x_4_d)
        de_x_2_5 = self.de_conv_2_5(x5_d)

        de_x_2_cat = torch.cat([de_x_2_1, de_x_2_2, de_x_2_3, de_x_2_4, de_x_2_5], 1)
        de_x_2_d = self.de_conv2_fusion(de_x_2_cat)

        # Decoding x_1
        de_x_1_1 = self.de_conv_1_1(x1)
        de_x_1_2 = self.de_conv_1_2(de_x_2_d)
        de_x_1_3 = self.de_conv_1_3(de_x_3_d)
        de_x_1_4 = self.de_conv_1_4(de_x_4_d)
        de_x_1_5 = self.de_conv_1_5(x5_d)

        de_x_1_cat = torch.cat([de_x_1_1, de_x_1_2, de_x_1_3, de_x_1_4, de_x_1_5], 1)
        de_x_1_cat = self.de_conv1_fusion(de_x_1_cat)

        output = self.outconv(de_x_1_cat)

        return output


model = UNet3Plus(in_dims=3, n_classes=1)
summary(model, (3, 224, 224), device='cpu')