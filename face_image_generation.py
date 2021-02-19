import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from networks.trainer_interface import *
from frameworks.gans_trainer import GansTrainer, GansModule
from utils.path_dataset import PathDataset
import pickle

class Runner:
    def __init__(
        self, 
        root_datapath = "./grid-dataset/samples/",
        output_path = "./trainer_output/",
        lr = 0.001,
        epochs = 10,
    ):
        self.__rootpath = root_datapath
        self.__output_path = output_path
        self.__dataloader = None
        self.__create_dataloader()
        self.__trainer = GansTrainer(epochs = epochs)

        self.__trainer.inject_train_dataloader(self.__produce_data)

        generator = GeneratorTrainerInterface()
        generator_module = GansModule (
            model = generator,
            optim = optim.Adam(generator.parameters(), lr = lr),
            loss_function = None,
        )
        self.__trainer.inject_generator(generator_module)

        sync_dis = SyncDiscriminatorTrainerInterface()
        seq_dis = SequenceDiscriminatorTrainerInterface()
        frame_dis = FrameDiscriminatorTrainerInterface()
        for name, dis in [
            ("sync_dis", sync_dis),
            ("seq_dis", seq_dis),
            ("frame_dis", frame_dis),
        ]:
            dis_module = GansModule(
                model = dis,
                optim = optim.Adam(dis.parameters(), lr = lr),
                loss_function = nn.BCELoss(),
            )
            self.__trainer.inject_discriminator(name, dis_module)

        self.__trainer.inject_other_loss_function("l1_loss", VideoL1Loss())
        self.__device = self.__trainer.get_device()
        
    def __create_dataloader(self):
        data_paths = list()
        for path, _ , files in os.walk(self.__rootpath):
            for name in files:
                ext = name.split('.')[-1]
                if ext != 'pkl':
                    continue
                data_paths.append(os.path.join(path, name))

        def data_processing(fd):
            data = pickle.load(fd)
            video = data['video']
            video = np.transpose(video, (1, 0, 2, 3))
            video = torch.tensor(video/255.0).float()
            audio = data['audio']
            audio = np.pad(audio, (486, 486), 'constant', constant_values = (0, 0))
            audio = torch.tensor(np.expand_dims(audio, axis = 0)).float()
            return (video, audio)

        dataset = PathDataset(data_paths, data_processing)
        params = {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True,
        }
        self.__dataloader = DataLoader(dataset, **params)

    def __produce_data(self):
        for image, audio in self.__dataloader:
            image = image.to(self.__device)
            audio = audio.to(self.__device)
            yield (image, audio)

    def start(self):
        self.__trainer.train()

def main():
    runner = Runner()
    runner.start()

if __name__ == "__main__":
    main()
