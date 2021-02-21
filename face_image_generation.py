import sys, os
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from networks.trainer_interface import *
from frameworks.gans_trainer import GansTrainer, GansModule
from utils.path_dataset import PathDataset
import pickle
from loguru import logger as log

class Runner:
    def __init__(
        self,
        device = "cpu",
        root_datapath = "./grid-dataset/samples/",
        output_path = "./trainer_output/",
        lr = 0.001,
        epochs = 10,
    ):
        self.__rootpath = root_datapath
        self.__output_path = output_path
        self.__dataloader = None
        self.__create_dataloader()
        self.__trainer = GansTrainer(epochs = epochs, device = device)

        self.__trainer.inject_train_dataloader(self.__produce_data)
        self.__trainer.inject_test_dataloader(self.__produce_data)
        self.__trainer.inject_evaluation_callback(self.__save_evaluation_data)
        self.__trainer.inject_save_model_callback(self.__save_model)

        generator = GeneratorTrainerInterface(device = device)
        generator_module = GansModule (
            model = generator,
            optim = optim.Adam(generator.parameters(), lr = lr),
            loss_function = None,
        )
        self.__trainer.inject_generator(generator_module)

        sync_dis = SyncDiscriminatorTrainerInterface(device = device)
        seq_dis = SequenceDiscriminatorTrainerInterface(device = device)
        frame_dis = FrameDiscriminatorTrainerInterface(device = device)
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

        self.__trainer.inject_other_loss_function("l1_loss", VideoL1Loss(device = device))
        self.__device = device
        
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
            audio = data['audio']/32768.0
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

    def __save_model(self, epoch, generator, discriminator_map):
        log.info(f"saving model for epoch {epoch}")
        # models_folder = "models"
        # epoch_folder = "epoch_{}".format(epoch)
        # folder_path = os.path.join(self.__output_path, models_folder, epoch_folder)
        # Path(folder_path).mkdir(parents=True, exist_ok=True)
        # generator_path = os.path.join(folder_path, "generator.pt")
        # torch.save(generator.state_dict(), generator_path)

        # for k, v in discriminator_map.items():
        #     dis_path = os.path.join(folder_path, k + '.pt')
        #     torch.save(v.state_dict(), dis_path)

    def __save_evaluation_data(self, epoch, sample, orig_data, generated_data):
        if sample != 0:
            return
        # data_folder = "data"
        # folder_path = os.path.join(self.__output_path, data_folder)
        # Path(folder_path).mkdir(parents=True, exist_ok=True)
        # data = {
        #     'orig_data': orig_data,
        #     'generated_data': generated_data,
        # }
        # data_path = os.path.join(folder_path, "data_{}.pkl".format(epoch))
        # with open(data_path, 'wb') as fd:
        #     pickle.dump(data, fd)

    def start(self):
        self.__trainer.train()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runner = Runner(device = device)
    runner.start()

if __name__ == "__main__":
    main()
