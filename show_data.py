import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import cv2
import ffmpeg
import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
from array2gif import write_gif
from utils.media import vidwrite
import scipy.io.wavfile as wav

def __show_images(video):
    batchsize = video.shape[0]
    video = np.transpose(video, (0,2,3,4,1))

    for k in range(batchsize):
        vid = video[k]
        fig, axis = plt.subplots(5, 15)
        for i in range(5):
            for j in range(15):
                img = vid[i*15+j,:,:,:]
                axis[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

def __gen_videos(videos):
    batchsize = videos.shape[0]
    videos = np.transpose(videos, (0,2,3,4,1))
    for k in range(batchsize):
        out = videos[k]
        for i in range(out.shape[0]):
            out[i] = cv2.cvtColor(out[i], cv2.COLOR_BGR2RGB)
        vidwrite("./output_{}_video.mp4".format(k), out, vcodec='libx264', fps=25)

def __show_models(rootpath='./model'):
    models = [
        'frame_dis.pt', 
        # 'seq_dis.pt', 
        # 'sync_dis.pt', 
        # 'generator.pt',
    ]
    for model in models:
        path = os.path.join(rootpath, model)
        state_dict = torch.load(path)
        print(path)
        print(state_dict)

def __gen_audios(audios):
    batchsize = audios.shape[0]
    for k in range(batchsize):
        out = audios[k][0]
        wav.write("./output_{}_audio.wav".format(k), 44100, out)

def main():
    path = sys.argv[1]
    mp = None
    with open(path, 'rb') as fd:
        mp = pickle.load(fd)
        print(mp.keys())
    orig_data = mp['orig_video']
    generated_data = mp['generated_data']
    combined = np.concatenate((orig_data, generated_data), axis = -1)
    audio = mp['audio']

    # __show_images(orig_data)
    # __show_images(generated_data)
    __gen_videos(combined)
    __gen_audios(audio)
    batchsize = audio.shape[0]
    for k in range(batchsize):
        v = ffmpeg.input("./output_{}_video.mp4".format(k))
        a = ffmpeg.input("./output_{}_audio.wav".format(k))
        out = ffmpeg.output(v['v'], a['a'], "./final_{}.mp4".format(k), loglevel="panic")
        out.run()
    # __show_models()


if __name__ == "__main__":
    main()
