import torch
import torch.nn as nn

from torchsummary.torchsummary import summary
from Model.UnetParts import DoubleConv


class NestedUNet(nn.Module):
    def __init__(self, in_dims=3, n_classes=1, deep_supervision=False):
        super(NestedUNet, self).__init__()

        num_filters = [64, 128, 256, 512, 1024]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_dims, num_filters[0], num_filters[0])
        self.conv1_0 = DoubleConv(num_filters[0], num_filters[1], num_filters[1])
        self.conv2_0 = DoubleConv(num_filters[1], num_filters[2], num_filters[2])
        self.conv3_0 = DoubleConv(num_filters[2], num_filters[3], num_filters[3])
        self.conv4_0 = DoubleConv(num_filters[3], num_filters[4], num_filters[4])

        # Upsampling & Dense skip
        # N to 1 skip
        self.conv0_1 = DoubleConv(num_filters[0] + num_filters[1], num_filters[0], num_filters[0])
        self.conv1_1 = DoubleConv(num_filters[1] + num_filters[2], num_filters[1], num_filters[1])
        self.conv2_1 = DoubleConv(num_filters[2] + num_filters[3], num_filters[2], num_filters[2])
        self.conv3_1 = DoubleConv(num_filters[3] + num_filters[4], num_filters[3], num_filters[3])

        # N to 2 skip
        self.conv0_2 = DoubleConv(num_filters[0] * 2 + num_filters[1], num_filters[0], num_filters[0])
        self.conv1_2 = DoubleConv(num_filters[1] * 2 + num_filters[2], num_filters[1], num_filters[1])
        self.conv2_2 = DoubleConv(num_filters[2] * 2 + num_filters[3], num_filters[2], num_filters[2])

        # N to 3 skip
        self.conv0_3 = DoubleConv(num_filters[0] * 3 + num_filters[1], num_filters[0], num_filters[0])
        self.conv1_3 = DoubleConv(num_filters[1] * 3 + num_filters[2], num_filters[1], num_filters[1])

        # N to 4 skip
        self.conv0_4 = DoubleConv(num_filters[0] * 4 + num_filters[1], num_filters[0], num_filters[0])

        if self.deep_supervision:
            self.output1 = nn.Conv2d(num_filters[0], n_classes, kernel_size=1)
            self.output2 = nn.Conv2d(num_filters[0], n_classes, kernel_size=1)
            self.output3 = nn.Conv2d(num_filters[0], n_classes, kernel_size=1)
            self.output4 = nn.Conv2d(num_filters[0], n_classes, kernel_size=1)

        else:
            self.output = nn.Conv2d(num_filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.output1(x0_1)
            output2 = self.output2(x0_2)
            output3 = self.output3(x0_3)
            output4 = self.output4(x0_4)
            output = (output1 + output2 + output3 + output4) / 4
        else:
            output = self.output(x0_4)

        return output

model = NestedUNet(in_dims=3, n_classes=1, deep_supervision=True)
summary(model, (3, 224, 224), device='cpu')