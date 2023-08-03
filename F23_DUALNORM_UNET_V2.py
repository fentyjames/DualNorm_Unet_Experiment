import torch
import torch.nn as nn
from F_UNET_PARTS import ResBlock, InConv, Down, Mid, Up, TransConv, OutConv

try:
  import torchinfo
except:
  !pip install torchinfo
  import torchinfo

from torchinfo import summary


class SigmoidOut(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SigmoidOut, self).__init__()
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x

class DualNorm_Unet_V2(nn.Module):
    def __init__(self, n_channels, n_classes, nonlinear='relu', dropout_rate=0.3):
        super(DualNorm_Unet_V2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        self.inc = InConv(n_channels, 32, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.down1 = Down(32, 64, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.down2 = Down(64, 128, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.down3 = Down(128, 256, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.down4 = Down(256, 512, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.mid = Mid(512, 512, n_classes=n_classes, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.up1 = Up(512, 256, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.up2 = Up(256, 128, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.up3 = Up(128, 64, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.up4 = Up(64, 32, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.outc = OutConv(32, n_classes, nonlinear=nonlinear, dropout_rate=dropout_rate)
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5_1 = self.down4(x4)
        x = self.mid(x5_1)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.outsig(x)
        return x
      


model_1 = DualNorm_Unet_V2(n_channels=3, n_classes=1, dropout_rate=0.0)  # Create an instance of the DualNorm_Unet class
summary(model_1,
        input_size=[(16, 3, 224, 224)],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )
