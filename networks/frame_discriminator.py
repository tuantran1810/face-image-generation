import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv2dBlock

class FrameDiscriminator(nn.Module):
    def __init__(self, input_channels = 3, start_hidden_channels = 32, hidden_layers = 5):
        super(FrameDiscriminator, self).__init__()
        kernel = (4, 4)
        stride = 2
        padding = 1
        w = 96
        h = 128
        channels = start_hidden_channels

        layers = [
            Conv2dBlock(
                in_channels = input_channels,
                out_channels = start_hidden_channels,
                kernel = kernel,
                stride = stride,
                padding = padding,
            ),
        ]
        w //= 2
        h //= 2

        for _ in range(hidden_layers):
            layers.append(
                Conv2dBlock(
                    in_channels = channels,
                    out_channels = channels*2,
                    kernel = kernel,
                    stride = stride,
                    padding = padding,
                ),
            )
            channels *= 2
            w //= 2
            h //= 2
            if w < kernel[0] or h < kernel[1]:
                kernel = (min(w, kernel[0]), min(h, kernel[1]))
                padding = 0

        if kernel != (4, 4):
            w = 1
            h = 1

        self.__fc =nn.Linear(in_features = channels * w * h, out_features = 1)
        self.__layers = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x: (batch, channels, w, h)
        output: (batch, prob)
        '''
        batch_size = x.shape[0]
        x = self.__layers(x)
        x = x.view(batch_size, -1)
        x = self.__fc(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    fd = FrameDiscriminator()
    x = torch.ones(30, 3, 96, 128)
    y = fd(x)
    print(y.shape)
