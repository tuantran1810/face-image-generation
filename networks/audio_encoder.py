import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from nets import Conv1dBlock

class AudioEncoder(nn.Module):
    def __init__(self, input_dims = 8000, input_channels = 1, output_dims = 256, hidden_layers = 4, device = "cpu"):
        super(AudioEncoder, self).__init__()
        channels = 16
        layers = [
            Conv1dBlock(
                in_channels = input_channels, 
                out_channels = channels, 
                kernel = 250, 
                stride = 50,
                padding = 124,
            )
        ]

        for _ in range(hidden_layers - 1):
            layers.append(
                Conv1dBlock(
                    in_channels = channels, 
                    out_channels = channels * 2, 
                    kernel = 4, 
                    stride = 2,
                    padding = 1,
                ),
            )
            channels *= 2

        layers.append(
            Conv1dBlock(
                in_channels = channels, 
                out_channels = channels * 2, 
                kernel = 10, 
                stride = 5,
                padding = 3,
            ),
        )
        channels *= 2

        final_layer = nn.Sequential(
            nn.Conv1d(
                in_channels = channels, 
                out_channels = channels, 
                kernel_size = 4, 
                stride = 1,
            ),
            nn.Tanh(),
        )
        layers.append(final_layer)
        self.__layers = nn.Sequential(*layers).to(device)
        self.__device = device

    def forward(self, x):
        '''
        x: (batch_size, channels, data)
        output: (batch_size, encoded_data)
        '''
        x = x.to(self.__device)
        return self.__layers(x).squeeze(2)

if __name__ == "__main__":
    x = torch.ones(3, 1, 8820)
    aenc = AudioEncoder(device = "cpu")
    print(aenc)
    print(aenc(x).shape)
