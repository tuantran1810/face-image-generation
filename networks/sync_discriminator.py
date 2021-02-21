import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoder
from mouth_sequence_encoder import MouthSequenceEncoder

class SyncDiscriminator(nn.Module):
    def __init__(self, device = "cpu"):
        super(SyncDiscriminator, self).__init__()
        self.__aenc = AudioEncoder(device = device)
        self.__menc = MouthSequenceEncoder(device = device)
        self.__fc = nn.Linear(256, 1).to(device)
        self.__device = device

    def forward(self, images, audio):
        '''
        images: (batch, channels, t, w, h)
        audio: (batch, channels, values)
        output: (batch, values)
        '''
        images = images.to(self.__device)
        audio = audio.to(self.__device)
        if images.shape[0] != audio.shape[0]:
            raise Exception("batch size of images and audio are not consistent")
        audio_feature = self.__aenc(audio)
        mouth_feature = self.__menc(images)
        x = torch.subtract(audio_feature, mouth_feature)
        x = torch.square(x)
        x = self.__fc(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    s_dis = SyncDiscriminator(device = "cpu")
    images = torch.ones(7, 3, 5, 96, 64)
    audio = torch.ones(7, 1, 8820)
    y = s_dis(images, audio)
    print(y.shape)
