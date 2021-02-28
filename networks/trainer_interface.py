import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn
from torch.autograd import Variable
from frame_discriminator import FrameDiscriminator
from sequence_discriminator import SequenceDiscriminator
from sync_discriminator import SyncDiscriminator
from generator import Generator

class GeneratorTrainerInterface(Generator):
    def __init__(self, device = "cpu"):
        super().__init__(device = device)
        self.__device = device

    def forward(self, orig_data):
        '''
        data: [batch_image, batch_audio]
        batch_image: (batch, channels, t, w, h)
        batch_audio: (batch, channels, values)
        output: (batch, channels, t, w, h)
        '''
        batch_image, batch_audio = orig_data
        batch_image = batch_image.to(self.__device)
        batch_audio = batch_audio.to(self.__device)
        batch_image = batch_image[:,:,0,:,:]
        batch_image = batch_image.squeeze(2)
        return super().forward(batch_image, batch_audio)

    def suggested_generator_training_label(self, xhat):
        raise NotImplementedError()

    def suggested_discriminator_training_label(self, xhat):
        raise NotImplementedError()

class SyncDiscriminatorTrainerInterface(SyncDiscriminator):
    def __init__(self, frame_stride = 5, device = "cpu"):
        self.__frame_stride = frame_stride
        self.__device = device
        super().__init__(device = device)

    def __aligned_audio_video_pairs(self, video, audio):
        return self.__missaligned_audio_video_pairs(video, audio, offset = 0)

    def __missaligned_audio_video_pairs(self, video, audio, offset = 1):
        video_shape = video.shape
        frames_per_video = video_shape[2]
        t_audio = audio.shape[-1]

        samples_per_video = frames_per_video//self.__frame_stride
        audio_stride = (t_audio//frames_per_video)*self.__frame_stride

        tmp_audio_arr = []
        tmp_video_arr = []
        for i in range(samples_per_video):
            j = (i+offset)%samples_per_video
            tmp_audio = audio[:,:,(j*audio_stride):((j+1)*audio_stride)]
            tmp_video = video[:,:,(i*self.__frame_stride):((i+1)*self.__frame_stride),:,:]
            tmp_audio_arr.append(tmp_audio)
            tmp_video_arr.append(tmp_video)
        
        return torch.cat(tmp_video_arr, 0), torch.cat(tmp_audio_arr, 0)

    def forward(self, orig_data, generated_data, discriminator_training):
        '''
        generated_data: (batch, channels, t, w, h)
        orig_data: [batch_image, batch_audio]
        batch_image: (batch, channels, t, w, h)
        batch_audio: (batch, channels, values)
        output: (batch, prob)
        '''
        orig_video, audio = orig_data
        gen_video = generated_data

        orig_video = orig_video.to(self.__device)
        audio = audio.to(self.__device)
        gen_video = gen_video.to(self.__device)

        h = orig_video.shape[-1]
        orig_video = orig_video[:,:,:,:,h//2:]
        gen_video = gen_video[:,:,:,:,h//2:]

        if not discriminator_training:
            # super().eval()
            video, audio = self.__aligned_audio_video_pairs(gen_video, audio)
            xhat = super().forward(video, audio)
            return xhat.squeeze(1)

        super().train()
        batch_size = orig_video.shape[0]

        upper_video = orig_video
        middle_video = orig_video[:batch_size//2]
        lower_video = gen_video[batch_size//2:]

        upper_audio = audio
        middle_audio = audio[:batch_size//2]
        lower_audio = audio[batch_size//2:]

        upper_video, upper_audio = self.__aligned_audio_video_pairs(upper_video, upper_audio)
        middle_video, middle_audio = self.__missaligned_audio_video_pairs(middle_video, middle_audio)
        lower_video, lower_audio = self.__aligned_audio_video_pairs(lower_video, lower_audio)

        video = torch.cat([upper_video, middle_video, lower_video], 0)
        audio = torch.cat([upper_audio, middle_audio, lower_audio], 0)
        xhat = super().forward(video, audio)
        return xhat.squeeze(1)

    def suggested_generator_training_label(self, xhat):
        # return 1.0 vector (real)
        k = xhat.shape[0]
        return Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False).to(self.__device)

    def suggested_discriminator_training_label(self, xhat):
        # return upper half 0.0 vector (fake), lower half 1.0 (real)
        k = xhat.shape[0]//2
        upper = Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False)
        lower = Variable(torch.FloatTensor(k).fill_(0.0), requires_grad=False)
        return torch.cat([upper, lower], 0).to(self.__device)

class SequenceDiscriminatorTrainerInterface(SequenceDiscriminator):
    def __init__(self, device = "cpu"):
        super().__init__(device = device)
        self.__device = device

    def forward(self, orig_data, generated_data, discriminator_training):
        '''
        generated_data: (batch, channels, t, w, h)
        orig_data: [batch_image, batch_audio]
        batch_image: (batch, channels, t, w, h)
        batch_audio: (batch, channels, values)
        output: (batch, prob)
        '''
        orig_video, orig_audio = orig_data
        gen_video = generated_data

        orig_video = orig_video.to(self.__device)
        orig_audio = orig_audio.to(self.__device)
        gen_video = gen_video.to(self.__device)

        if not discriminator_training:
            # super().eval()
            xhat = super().forward(gen_video, orig_audio)
            return xhat.squeeze(1)

        video = torch.cat([gen_video, orig_video], 0)
        audio = torch.cat([orig_audio, orig_audio], 0)
        super().train()
        xhat = super().forward(video, audio)
        return xhat.squeeze(1)

    def suggested_generator_training_label(self, xhat):
        # return 1.0 vector (real)
        k = xhat.shape[0]
        return Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False).to(self.__device)

    def suggested_discriminator_training_label(self, xhat):
        # return upper half 0.0 vector (fake), lower half 1.0 (real)
        k = xhat.shape[0]//2
        upper = Variable(torch.FloatTensor(k).fill_(0.0), requires_grad=False).to(self.__device)
        lower = Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False).to(self.__device)
        return torch.cat([upper, lower], 0)

class FrameDiscriminatorTrainerInterface(FrameDiscriminator):
    def __init__(self, device = "cpu"):
        super().__init__(device = device)
        self.__device = device

    def forward(self, orig_data, generated_data, discriminator_training):
        '''
        generated_data: (batch, channels, t, w, h)
        orig_data: [batch_image, batch_audio]
        batch_image: (batch, channels, t, w, h)
        batch_audio: (batch, channels, values)
        output: (batch, prob)
        '''
        orig_video, _ = orig_data

        orig_video = orig_video.to(self.__device)
        generated_data = generated_data.to(self.__device)
        
        generated_data = generated_data.transpose(1, 2)
        shape = generated_data.shape
        generated_data = generated_data.reshape(-1, *shape[2:])

        if not discriminator_training:
            # super().eval()
            xhat = super().forward(generated_data)
            return xhat.squeeze(1)

        
        orig_video = orig_video.transpose(1, 2)
        shape = orig_video.shape
        orig_video = orig_video.reshape(-1, *shape[2:])

        data = torch.cat([generated_data, orig_video], 0)
        super().train()
        xhat = super().forward(data)
        return xhat.squeeze(1)

    def suggested_generator_training_label(self, xhat):
        # return 1.0 vector (real)
        k = xhat.shape[0]
        return Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False).to(self.__device)

    def suggested_discriminator_training_label(self, xhat):
        # return upper half 0.0 vector (fake), lower half 1.0 (real)
        k = xhat.shape[0]//2
        upper = Variable(torch.FloatTensor(k).fill_(0.0), requires_grad=False).to(self.__device)
        lower = Variable(torch.FloatTensor(k).fill_(1.0), requires_grad=False).to(self.__device)
        return torch.cat([upper, lower], 0)

class VideoL1Loss(nn.Module):
    def __init__(self, weight = 1.0, device = "cpu"):
        super(VideoL1Loss, self).__init__()
        super().to(device)
        self.__device = device
        self.__weight = weight

    def forward(self, orig_data, generated_data):
        orig_video, _ = orig_data
        orig_video = orig_video.to(self.__device)
        generated_data = generated_data.to(self.__device)

        h = orig_video.shape[-1]
        orig_video = orig_video[:,:,:,:,h//2:]
        generated_data = generated_data[:,:,:,:,h//2:]
        return nn.functional.l1_loss(generated_data, orig_video)*self.__weight

class WeightedBCELoss(nn.Module):
    def __init__(self, weight = 1.0, device = "cpu"):
        super(WeightedBCELoss, self).__init__()
        super().to(device)
        self.__device = device
        self.__weight = weight

    def forward(self, xhat, target):
        xhat = xhat.to(self.__device)
        target = target.to(self.__device)
        loss = nn.functional.l1_loss(xhat, target) * self.__weight
        return loss

if __name__ == "__main__":
    device = "cuda:0"
    gen = GeneratorTrainerInterface(device = device)
    sync = SyncDiscriminatorTrainerInterface(device = device)
    seq = SequenceDiscriminatorTrainerInterface(device = device)
    frame = FrameDiscriminatorTrainerInterface(device = device)

    images = torch.randn(4, 3, 75, 96, 128).to(device)
    audios = torch.randn(4, 1, 132300).to(device)
    orig_data = (images, audios)

    gen_data = gen(orig_data)
    print(f"gen shape: {images.shape}")

    all_loss = []

    xhat = sync(orig_data, gen_data, False)
    y = sync.suggested_generator_training_label(xhat)
    print(f"sync-infer shape: {xhat.shape}")
    loss = nn.functional.binary_cross_entropy(xhat, y)
    all_loss.append(loss)

    xhat = seq(orig_data, gen_data, False)
    y = seq.suggested_generator_training_label(xhat)
    print(f"seq-infer shape: {xhat.shape}")
    loss = nn.functional.binary_cross_entropy(xhat, y)
    all_loss.append(loss)

    xhat = frame(orig_data, gen_data, False)
    y = seq.suggested_generator_training_label(xhat)
    print(f"frame-infer shape: {xhat.shape}")
    loss = nn.functional.binary_cross_entropy(xhat, y)
    all_loss.append(loss)

    loss = nn.functional.l1_loss(gen_data, images)
    all_loss.append(loss)
    
    total_loss = sum(all_loss)
    print(f"total loss: {total_loss}")
    total_loss.backward()

    gen_data = gen_data.detach()
    xhat = sync(orig_data, gen_data, True)
    y = sync.suggested_discriminator_training_label(xhat)
    print(f"sync-train shape: {xhat.shape}")
    sync_loss = nn.functional.binary_cross_entropy(xhat, y)
    sync_loss.backward(retain_graph=True)

    xhat = seq(orig_data, gen_data, True)
    y = seq.suggested_discriminator_training_label(xhat)
    print(f"seq-train shape: {xhat.shape}")
    seq_loss = nn.functional.binary_cross_entropy(xhat, y)
    seq_loss.backward(retain_graph=True)

    xhat = frame(orig_data, gen_data, True)
    y = frame.suggested_discriminator_training_label(xhat)
    print(f"frame-train shape: {xhat.shape}")
    frame_loss = nn.functional.binary_cross_entropy(xhat, y)
    frame_loss.backward(retain_graph=True)
