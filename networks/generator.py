import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_rnn import AudioRNN
from identity_encoder import IdentityEncoder
from frame_decoder import FrameDecoder
from noise_rnn import NoiseRNN

class Generator(nn.Module):
    def __init__(self, device = "cpu"):
        super(Generator, self).__init__()
        self.__audio_rnn = AudioRNN(device = device)
        self.__identity_enc = IdentityEncoder(device = device)
        self.__noise_enc = NoiseRNN(device = device)
        self.__frame_dec = FrameDecoder(device = device)
        self.__device = device

    def forward(self, images, audios):
        '''
        images: (batch, channels, w, h)
        audios: (batch, channels, data)
        output: (batch, channels, sequence, w, h)
        '''
        images = images.to(self.__device)
        audios = audios.to(self.__device)

        batch_size = images.shape[0]
        if images.shape[0] != audios.shape[0]:
            raise Exception("batch size of images and audios are not consistent")
        id_feature, skip_features = self.__identity_enc(images)
        audio_feature = self.__audio_rnn(audios)
        t = audio_feature.shape[1]
        noise = torch.randn(batch_size, t, 10)
        noise_feature = self.__noise_enc(noise)

        total_batch = t * batch_size
        id_feature = id_feature.unsqueeze(1).repeat(1, t, 1).reshape(total_batch, -1)
        audio_feature = audio_feature.reshape(total_batch, -1)
        noise_feature = noise_feature.reshape(total_batch, -1)
        new_skip_features = []
        for fea in skip_features:
            old_size = fea.shape
            fea = fea.unsqueeze(1).repeat(1, t, 1, 1, 1).reshape(total_batch, *old_size[1:])
            new_skip_features.append(fea)
        skip_features = tuple(new_skip_features)
        output = self.__frame_dec(id_feature, audio_feature, noise_feature, skip_features)
        output_shape = output.shape
        output = output.reshape(batch_size, t, *output_shape[1:])
        output = output.transpose(1, 2)
        return output

if __name__ == "__main__":
    gen = Generator(device = "cpu")
    images = torch.randn(2, 3, 96, 128)
    audios = torch.randn(2, 1, 132300)
    print(gen)
    out = gen(images, audios)
    print(out.shape)
