import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv3dBlock, Conv2dBlock

class MouthSequenceEncoder(nn.Module):
    def __init__(self, input_channels = 3, sequence_length = 5, start_hidden_channels = 64, out_features = 256, device = "cpu"):
        super(MouthSequenceEncoder, self).__init__()
        channels = start_hidden_channels
        kernel = (4, 4)
        stride = 2
        padding = 1
        w = 96
        h = 64
        self.__gate = Conv3dBlock(
            in_channels = input_channels,
            out_channels = channels,
            kernel = (sequence_length, *kernel),
            stride = (1, stride, stride),
            padding = (0, padding, padding),
        ).to(device)
        w //= 2
        h //= 2

        layers = []
        while True:
            layers.append(
                Conv2dBlock(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel = kernel,
                    stride = stride,
                    padding = padding,
                )
            )
            channels *= 2
            w //= 2
            h //= 2
            if w < kernel[0] or h < kernel[1]:
                kernel = (min(w, kernel[0]), min(h, kernel[1]))
                padding = 0
                break
        self.__layers = nn.Sequential(*layers).to(device)
        self.__output = nn.Sequential(
            Conv2dBlock(
                in_channels = channels,
                out_channels = out_features,
                kernel = kernel,
                stride = stride,
                padding = padding,
            ),
            nn.Tanh(),
        ).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batch, channels, t, w, h)
        output: (batch, values)
        '''
        x = x.to(self.__device)
        x = self.__gate(x)
        x = x.squeeze(2)
        x = self.__layers(x)
        x = self.__output(x)
        x = x.squeeze(3).squeeze(2)
        return x

if __name__ == "__main__":
    m_enc = MouthSequenceEncoder(device = "cpu")
    x = torch.ones(7, 3, 5, 96, 64)
    y = m_enc(x)
    print(y.shape)
