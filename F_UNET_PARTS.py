import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, nonlinear='relu', dropout_rate=0.3):
        super(ResBlock, self).__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.norm_0 = nn.BatchNorm2d(out_channels)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)  # Dropout layer
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv_0(x)
        x = self.nonlinear_layer(x)
        x = self.norm_0(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv_1(x)
        x = self.nonlinear_layer(x)
        x = self.norm_1(x)
        x += residual
        return x


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, block=ResBlock, batchsize=4, nonlinear='relu',dropout_rate=0.0):
        super(InConv, self).__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )
        self.block = ResBlock(out_channels, out_channels, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        #print("Print Input shape: ", x.size())
        return x



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, block=ResBlock, batchsize=4, nonlinear='relu',dropout_rate=0.2):
        super(Down, self).__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #kernel_size=2, stride=2
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1) #stride=1,,
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )
        self.block = ResBlock(out_channels, out_channels, n_classes)

    def forward(self, x):
        x = self.maxpool_conv(x)
        #print("Print Maxpool: ", x.size())
        x = self.block(x)
        return x


class Mid(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, batchsize=4, nonlinear='relu',dropout_rate=0.0):
        super(Mid, self).__init__()
        self.block = ResBlock(in_channels, out_channels, n_classes)

    def forward(self, x):
        x = self.block(x)
        return x



class TransConv(nn.Module):
    def __init__(self, in_channels, batchsize=4, nonlinear='relu',dropout_rate=0.0):
        super(TransConv, self).__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channels), #(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, n_classes=2, bilinear=True, block=ResBlock, batchsize=4,
                 nonlinear='relu',dropout_rate=0.3 ):
        super().__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.up = TransConv(in_channels, batchsize=batchsize, nonlinear=nonlinear, )
        self.conv = nn.Sequential(
                #nn.BatchNorm2d(out_channels),
                #nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        self.blockname = block
        self.block = block(out_channels, out_channels, n_classes=n_classes,
                               nonlinear=nonlinear, )

    def forward(self, x1, x2, seg=None):
        x1 = self.up(x1)
        #print("Print Up X1: ", x1.size())
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        #print("The values of X111", x.size())
        x = self.conv(x)
        x = self.block(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchsize=4, nonlinear='relu',dropout_rate=0.0):
        super(OutConv, self).__init__()
        self.nonlinear_layer = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1) #, stride=1
        )

    def forward(self, x):
        x = self.conv(x)
        #x = self.nonlinear_layer(x)
        #print("Final input X", x.size())
        return x
