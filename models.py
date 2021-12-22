import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import data
import numpy as np
from config import config as cfg

# Unet的下采样模块，两次卷积
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.Conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),stride=1, padding=1)
        self.BatchNorm2d_1 = nn.BatchNorm2d(out_channels)
        self.ReLU = F.relu
        self.Conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),stride=1, padding=1)
        self.BatchNorm2d_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.BatchNorm2d_1(x)
        x = self.ReLU(x)

        x = self.Conv2d_2(x)
        x = self.BatchNorm2d_2(x)
        x = self.ReLU(x)
        return x


# 上采样（转置卷积加残差链接）
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_conv = self.conv(x)
        x_down = self.down(x_conv)
        return x_conv,x_down


class U_net(nn.Module):
    def __init__(self,n = 64):
        super(U_net, self).__init__()

        self.down1 = Down(2, n)
        self.down2 = Down(n, 2*n)
        self.down3 = Down(2*n, 4*n)
        self.down4 = Down(4*n, 8*n)

        self.conv1 = DoubleConv(8*n, 16*n)

        self.up1 = Up(16*n,8*n)
        self.up2 = Up(8*n, 4*n)
        self.up3 = Up(4*n, 2*n)
        self.up4 = Up(2*n, n)

        self.out = nn.Conv2d(n, 4, kernel_size=(1, 1), padding=0)

    def forward(self,x):
        c1, d1 = self.down1(x)
        c2, d2 = self.down2(d1)
        c3, d3 = self.down3(d2)
        c4, d4 = self.down4(d3)
        c5 = self.conv1(d4)
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)
        return out

if __name__ == "__main__":
    test_img = cv2.imread("test.png",0)
    test_data = torch.from_numpy(test_img).unsqueeze(0).unsqueeze(0).float()
    model = U_net()
    out = model(test_data)
    out = out.cpu().detach().numpy()[0]
    points = data.heatmap_to_point(out,test_img.shape)

    result = data.show_result_on_img(test_img,points)
    cv2.imwrite("result.png",result)