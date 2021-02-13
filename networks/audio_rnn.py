import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from audio_encoder import AudioEncoder

class AudioRNN(nn.Module):
    def __init__(self, 
        sampling_rate = 44100,
        chunk_size = 8000,
        frame_rate = 25,
        output_dims = 256,
        hidden_gru_layers = 2,
        batch_mode = True,
    ):
        super(AudioRNN, self).__init__()
        self.__batch_mode = batch_mode
        self.__enc = AudioEncoder(input_dims = chunk_size, input_channels = 1, output_dims = output_dims)
        self.__gru = nn.GRU(input_size = output_dims, hidden_size = output_dims, num_layers = hidden_gru_layers, batch_first = True)
        self.__stride = sampling_rate//frame_rate
        padding_size = (chunk_size - self.__stride)//2
        self.__padding = nn.ZeroPad2d((padding_size, padding_size, 0, 0))
        self.__chunk_size = chunk_size

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
    
    def forward(self, x):
        '''
        x: (batch_size, channels, data)
        output: (batch_size, sequence, data)
        '''
        batch_size = x.shape[0]
        x = self.__padding1D(x)
        if self.__batch_mode:
            x = self.__audio_split(x)
            t_seq = x.shape[1]
            x = x.reshape(batch_size * t_seq, 1, -1)
            x = self.__enc(x)
            x = x.view(batch_size, t_seq, -1)
        else:
            chunks = [self.__enc(chunk) for chunk in self.__audio_chunks(x)]
            x = torch.stack(chunks)
            x = x.transpose(0, 1)

        out, _ = self.__gru(x)
        return out

if __name__ == "__main__":
    au_rnn = AudioRNN()
    print(au_rnn)
    x = torch.ones(30, 1, 132300)
    y = au_rnn(x)
    print(y.shape)
