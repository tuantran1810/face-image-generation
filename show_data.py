import sys, os
sys.path.append(os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

def main():
    path = sys.argv[1]
    mp = None
    with open(path, 'rb') as fd:
        mp = pickle.load(fd)
        print(mp.keys())
    # orig_data = mp['orig_data']
    generated_data = mp['generated_data']

    batchsize = generated_data.shape[0]
    generated_data = generated_data.cpu().numpy()
    generated_data = np.transpose(generated_data, (0,2,3,4,1))
    print(generated_data.shape)
    for k in range(batchsize):
        vid = generated_data[k]
        fig, axis = plt.subplots(5, 15)
        for i in range(5):
            for j in range(15):
                img = vid[i*15+j,:,:,:]
                axis[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

if __name__ == "__main__":
    main()
