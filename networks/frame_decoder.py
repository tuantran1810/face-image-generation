import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Deconv2dBlock, Unet2dBlock

class FrameDecoder(nn.Module):
    def __init__(self, input_channels = 394, start_hidden_channels = 1024, start_kernel = (3, 4), hidden_layers = 4, output_channels = 3):
        super(FrameDecoder, self).__init__()
        self.__gate = Deconv2dBlock(
            in_channels = input_channels,
            out_channels = start_hidden_channels,
            kernel = start_kernel,
        )
        channels = start_hidden_channels
        kernel = 4
        stride = 2
        padding = 1

        self.__hidden = []
        for _ in range(hidden_layers):
            self.__hidden.append(
                Unet2dBlock(
                    in_channels = channels,
                    skip_channels = channels,
                    out_channels = channels//2,
                    kernel = kernel,
                    stride = stride,
                    padding = padding,
                ),
            )
            channels //= 2
        
        self.__output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = channels,
                out_channels = output_channels,
                kernel_size = kernel,
                stride = stride,
                padding = padding,
            ),
            nn.Tanh(),
        )

    def forward(self, identity, audio, noise, forward_layers):
        '''
        identity, audio, noise: (batch, values)
        forward_layers: tuple of identity hidden layers (batch, channels, w, h)
        output: (batch, channels, w, h)
        '''
        if len(forward_layers) != len(self.__hidden):
            raise Exception("number of forward features and hidden layers are not consistent")
        n_hidden = len(self.__hidden)
        batch_size = identity.shape[0]
        x = torch.cat([identity, audio, noise], 1)
        x = x.view(*x.shape, 1, 1)
        x = self.__gate(x)

        for i, unet in enumerate(self.__hidden):
            s = forward_layers[n_hidden - 1 - i]
            x = unet(x, s)
        return self.__output_layer(x)

if __name__ == "__main__":
    fdec =   FrameDecoder()
    identity = torch.ones(30, 128)
    audio = torch.ones(30, 256)
    noise = torch.ones(30, 10)

    skip = [
        torch.ones(30, 128, 24, 32),
        torch.ones(30, 256, 12, 16),
        torch.ones(30, 512, 6, 8),
        torch.ones(30, 1024, 3, 4),
    ]

    y = fdec(identity, audio, noise, tuple(skip))
    print(y.shape)

