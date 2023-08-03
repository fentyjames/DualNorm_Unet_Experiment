import torch
import torch.nn as nn
from F_UNET_PARTS_V1 import ResBlock, InConv, Down, Mid, Up, TransConv, OutConv

try:
  import torchinfo
except:
  !pip install torchinfo
  import torchinfo

from torchinfo import summary


class DualNorm_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, nonlinear='relu'):
        super(DualNorm_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = InConv(n_channels, 32, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.down1 = Down(32, 64, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.down2 = Down(64, 128, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.down3 = Down(128, 256, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.down4 = Down(256, 512, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.mid = Mid(512, 512, n_classes=n_classes, nonlinear=nonlinear)
        self.up1 = Up(512, 256, n_classes=n_classes, block=ResBlock, nonlinear=nonlinear)
        self.up2 = Up(256, 128, n_classes=n_classes,  block=ResBlock, nonlinear=nonlinear)
        self.up3 = Up(128, 64, n_classes=n_classes,  block=ResBlock, nonlinear=nonlinear)
        self.up4 = Up(64, 32, n_classes=n_classes,  block=ResBlock, nonlinear=nonlinear)
        self.outc = OutConv(32, n_classes, nonlinear=nonlinear)
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



model_0 = DualNorm_Unet(n_channels=3, n_classes=1, )  # Create an instance of the DualNorm_Unet class dropout_rate=0.0
summary(model_0,
        input_size=[(16, 3, 224, 224)],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )
