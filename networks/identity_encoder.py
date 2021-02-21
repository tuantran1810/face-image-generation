import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv2dBlock

class IdentityEncoder(nn.Module):
    def __init__(self, input_channels = 3, start_hidden_channels = 64, hidden_layers = 4, output_channels = 128, device = "cpu"):
        super(IdentityEncoder, self).__init__()

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

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = channels,
                out_channels = output_channels,
                kernel_size = kernel,
                stride = stride,
                padding = 0,
            ),
            nn.Tanh(),
        )
        layers.append(final_layer)

        self.__layers = nn.Sequential(*layers).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batch_size, channels, w, h)
        output: (batch_size, vector) for encoded vector, (batch_size, channels, w, h) for hidden layers
        '''
        x = x.to(self.__device)
        batch_size = x.shape[0]
        lst = []
        for layer in self.__layers:
            x = layer(x)
            lst.append(x)
        output = lst[-1]
        output = output.view(batch_size, -1)
        return output, lst[1:-1]

if __name__ == "__main__":
    x = torch.ones(5, 3, 96, 128)
    i_enc = IdentityEncoder(device = "cpu")
    y, lst = i_enc(x)
    print(i_enc)
    for out in lst:
        print(out.shape)
    print(y.shape)