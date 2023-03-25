import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Sequential):
    """每次会进行两次卷积操作，故直接建立连续两次conv的sequential"""
    def __init__(self,in_channel,out_channel,middel_channel=None):
        if middel_channel is None:
            middel_channel = out_channel
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channel,middel_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(middel_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middel_channel,out_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self,in_channel,out_channel):
        super(Down, self).__init__(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channel=in_channel,out_channel=out_channel)
        )

class Up(nn.Module):
    def __init__(self,in_channel,out_channel,bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = DoubleConv(in_channel=in_channel,out_channel=out_channel,middel_channel=in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channel,out_channel)

    def forward(self,x1,x2):
        """
        @param x1: 上采样上来的
        @param x2: 捷径分支过来的
        """
        x1 = self.up(x1)
        # diff_h = x1.size()[2] - x2.size()[2]
        # diff_w = x1.size()[3] - x2.size()[3]
        #
        # x1 = F.pad(x1,[diff_w // 2,diff_w // 2,diff_h // 2,diff_h // 2])
        # print("下采样:",x1.shape,"捷径:",x2.shape)
        x = torch.concat((x1,x2),dim=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self,num_classes,in_channel,bilinear=True,basedim=64):
        super(Unet, self).__init__()
        self.dconv1 = DoubleConv(in_channel=in_channel,out_channel=basedim)
        self.down1 = Down(in_channel=basedim,out_channel=basedim * 2)
        self.down2 = Down(in_channel=basedim * 2,out_channel=basedim * 4)
        self.down3 = Down(in_channel=basedim * 4,out_channel=basedim * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(in_channel=basedim * 8,out_channel=basedim * 16 //factor )
        self.up1 = Up(in_channel=basedim * 16,out_channel=basedim * 8 // factor,bilinear=bilinear)
        self.up2 = Up(in_channel=basedim * 8,out_channel=basedim * 4 // factor,bilinear=bilinear)
        self.up3 = Up(in_channel=basedim * 4,out_channel=basedim * 2 // factor,bilinear=bilinear)
        self.up4 = Up(in_channel=basedim * 2,out_channel=basedim,bilinear=bilinear)

        self.out = nn.Conv2d(basedim,num_classes,kernel_size=1)

    def forward(self,x):
        x_c = self.dconv1(x)
        x_d1 = self.down1(x_c)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)

        x_u1 = self.up1(x_d4,x_d3)
        x_u2 = self.up2(x_u1,x_d2)
        x_u3 = self.up3(x_u2,x_d1)
        x_u4 = self.up4(x_u3,x_c)

        out = self.out(x_u4)
        return {"out":out}
