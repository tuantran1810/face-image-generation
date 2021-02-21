import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoder
from identity_encoder import IdentityEncoder

class SequenceDiscriminator(nn.Module):
    def __init__(
        self, 
        sampling_rate = 44100,
        chunk_size = 8000,
        frame_rate = 25,
        audio_output_dims = 256,
        frame_output_dims = 256,
        batch_mode = True,
        device = "cpu"
    ):
        super(SequenceDiscriminator, self).__init__()
        self.__batch_mode = batch_mode
        self.__stride = sampling_rate//frame_rate
        padding_size = (chunk_size - self.__stride)//2
        self.__padding = nn.ZeroPad2d((padding_size, padding_size, 0, 0)).to(device)
        total_dims = audio_output_dims + frame_output_dims
        self.__chunk_size = chunk_size

        self.__aenc = AudioEncoder(input_dims = chunk_size, input_channels = 1, output_dims = audio_output_dims, device = device)
        self.__fenc = IdentityEncoder(output_channels = frame_output_dims, device = device)
        self.__a_gru = nn.GRU(input_size = audio_output_dims, hidden_size = audio_output_dims, num_layers = 1, batch_first = True).to(device)
        self.__f_gru = nn.GRU(input_size = frame_output_dims, hidden_size = frame_output_dims, num_layers = 1, batch_first = True).to(device)
        self.__output_layers = nn.Sequential(
            nn.Linear(in_features = total_dims, out_features = total_dims//2),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = total_dims//2, out_features = 1),
            nn.Sigmoid(),
        ).to(device)
        self.__device = device

    def __padding1D(self, x):
        x = torch.unsqueeze(x, 2)
        x = self.__padding(x)
        x = torch.squeeze(x, 2)
        return x

    def __audio_chunks(self, x):
        audio_length = x.shape[2]
        for i in range(0, audio_length + 1 - self.__chunk_size, self.__stride):
            chunk = x[:,:,i:(i+self.__chunk_size)]
            yield chunk

    def __audio_split(self, x):
        chunks = [chunk for chunk in self.__audio_chunks(x)]
        y = torch.stack(chunks).transpose(0, 1)
        return y

    def forward(self, frames, audio):
        '''
        frames: (batch, channels, t, w, h)
        audio: (batch, channels, values)
        output: (batch, bools)
        '''
        frames = frames.to(self.__device)
        audio = audio.to(self.__device)
        if frames.shape[0] != audio.shape[0]:
            raise Exception("audio and frame batch size are not consistent")
        batch_size = audio.shape[0]
        audio = self.__padding1D(audio)
        if self.__batch_mode:
            audio = self.__audio_split(audio)
            t_seq = audio.shape[1]
            audio = audio.reshape(batch_size * t_seq, 1, -1)
            audio = self.__aenc(audio)
            audio = audio.view(batch_size, t_seq, -1)

            frames = frames.transpose(1, 2)
            total_frames = frames.shape[0]*frames.shape[1]
            frames = frames.reshape(total_frames, *frames.shape[2:])
            frames, _ = self.__fenc(frames)
            frames = frames.view(batch_size, -1, *frames.shape[1:])
        else:
            chunks = [self.__aenc(chunk) for chunk in self.__audio_chunks(audio)]
            audio = torch.stack(chunks)
            audio = audio.transpose(0, 1)

            frames_array = []
            for i in range(batch_size):
                fbatch = frames[i,:,:,:,:]
                fbatch = fbatch.transpose(0, 1)
                fbatch, _ = self.__fenc(fbatch)
                frames_array.append(fbatch)
            frames = torch.stack(frames_array, 0)

        _, audio = self.__a_gru(audio)
        audio = audio.squeeze(0)
        _, frames = self.__a_gru(frames)
        frames = frames.squeeze(0)

        combine = torch.cat([audio, frames], 1)
        combine = self.__output_layers(combine)
        return combine

if __name__ == "__main__":
    sd = SequenceDiscriminator(batch_mode = True, device = "cpu")
    audio = torch.ones(7, 1, 44100*3)
    frames = torch.ones(7, 3, 75, 96, 128)
    y = sd(frames, audio)
    print(y.shape)
