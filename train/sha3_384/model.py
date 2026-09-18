'Module object.'
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================

# ============================================================

class CyclicPad2d(nn.Module):
    def __init__(self, pad_z, pad_x, pad_y=1):
        super().__init__()
        self.pad_z = pad_z
        self.pad_x = pad_x
        self.pad_y = pad_y

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"expected 5D input, got {x.ndim}D")


        if self.pad_y > 0:
            x = F.pad(x, (0,0, 0,0, self.pad_y,self.pad_y), mode='constant', value=0)

        if self.pad_z > 0 or self.pad_x > 0:
            x = F.pad(x, (self.pad_x,self.pad_x, self.pad_z,self.pad_z, 0,0), mode='circular')

        return x


class CyclicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_z=3, kernel_x=3, kernel_y=1, stride=1):
        super().__init__()
        self.pad_z = kernel_z // 2
        self.pad_x = kernel_x // 2
        self.pad_y = kernel_y // 2
        self.padder = CyclicPad2d(self.pad_z, self.pad_x, self.pad_y)
        self.conv = nn.Conv3d(in_ch, out_ch,
                              kernel_size=(kernel_y, kernel_z, kernel_x),
                              stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.padder(x)
        return self.conv(x)
