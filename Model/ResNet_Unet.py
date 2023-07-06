import torch
from torchvision.models.resnet import resnet18 as _resnet18
import torch.nn as nn

from torchsummary.torchsummary import summary
# from Model.UnetParts import DoubleConv, DownDoubleConv, UpDoubleConv


class ResUNet(nn.Module):
    def __init__(self, backbone='resnet18', n_classes=1):
        super(ResUNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pool = self.model.maxpool
        self._first_conv = self._make_first_conv()

        self.res_block_1 = self.model.layer1
        self.res_block_2 = self.model.layer2
        self.res_block_3 = self.model.layer3
        self.res_block_4 = self.model.layer4

        self.res_block_4_1x1 = self.convrelu(512, 512, 3, 1)
        self.res_block_3_1x1 = self.convrelu(256, 256, 3, 1)
        self.res_block_2_1x1 = self.convrelu(128, 128, 3, 1)
        self.res_block_1_1x1 = self.convrelu(64, 64, 3, 1)
        self.first_conv_1x1 = self.convrelu(64, 64, 3, 1)

        self.conv_relu_4 = self.convrelu(256 + 512, 512, 3, 1)
        self.conv_relu_3 = self.convrelu(128 + 512, 256, 3, 1)
        self.conv_relu_2 = self.convrelu(64 + 256, 256, 3, 1)
        self.conv_relu_1 = self.convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = self.convrelu(3, 64, 3, 1)
        self.conv_original_size1 = self.convrelu(64, 64, 3, 1)
        self.conv_original_size2 = self.convrelu(64 + 128, 64, 3, 1)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=(1, 1))

    def _make_first_conv(self):
        module = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
        )

        return module

    def convrelu(self, in_dims, out_dims, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel, padding=padding),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        x0 = self._first_conv(x)
        x1 = self.pool(x0)
        x1 = self.res_block_1(x1)
        x2 = self.res_block_2(x1)
        x3 = self.res_block_3(x2)
        x4 = self.res_block_4(x3)

        x4 = self.res_block_4_1x1(x4)
        x4_up = self.up(x4)
        x3 = self.res_block_3_1x1(x3)
        x_cat = torch.cat([x4_up, x3], dim=1)
        x_cat = self.conv_relu_4(x_cat)

        x_up = self.up(x_cat)
        x2 = self.res_block_2_1x1(x2)
        x_cat = torch.cat([x_up, x2], dim=1)
        x_cat = self.conv_relu_3(x_cat)

        x_up = self.up(x_cat)
        x1 = self.res_block_1_1x1(x1)
        x_cat = torch.cat([x_up, x1], dim=1)
        x_cat = self.conv_relu_2(x_cat)

        x_up = self.up(x_cat)
        x0 = self.first_conv_1x1(x0)
        x_cat = torch.cat([x_up, x0], dim=1)
        x_cat = self.conv_relu_1(x_cat)

        x_up = self.up(x_cat)
        x = torch.cat([x_up, x_original], dim=1)
        x = self.conv_original_size2(x)

        output = self.outconv(x)

        return output


model = ResUNet()
summary(model, (3, 224, 224), device='cpu')
