import sys, os
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import torch
from torch import nn, optim
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader
import numpy as np
from networks.trainer_interface import *
from frameworks.gans_trainer import GansTrainer, GansModule
from utils.path_dataset import PathDataset
import pickle
from loguru import logger as log
from matplotlib import pyplot as plt

class Runner:
    def __init__(
        self,
        device = "cpu",
        train_datapath = "/media/tuantran/rapid-data/dataset/GRID/face_images",
        test_datapath = "./grid-dataset/samples/",
        output_path = "./trainer_output/",
        pretrained_model_paths = dict(),
        batchsize = 2,
        epochs = 10,
    ):
        self.__output_path = output_path
        self.__train_dataloader = self.__create_dataloader(train_datapath, batchsize)
        self.__test_dataloader = self.__create_dataloader(test_datapath, batchsize)
        
        self.__trainer = GansTrainer(epochs = epochs, device = device)

        self.__trainer.inject_train_dataloader(self.__produce_train_data)
        self.__trainer.inject_test_dataloader(self.__produce_test_data)
        self.__trainer.inject_evaluation_callback(self.__save_evaluation_data)
        self.__trainer.inject_save_model_callback(self.__save_model)

        generator = GeneratorTrainerInterface(device = device)
        if 'generator' in pretrained_model_paths:
            path = pretrained_model_paths['generator']
            generator.load_state_dict(torch.load(path))

        generator_module = GansModule (
            model = generator,
            optim = optim.Adam(generator.parameters(), lr = 0.0001),
            loss_function = None,
        )
        self.__trainer.inject_generator(generator_module)

        sync_dis = SyncDiscriminatorTrainerInterface(device = device)
        seq_dis = SequenceDiscriminatorTrainerInterface(device = device)
        frame_dis = FrameDiscriminatorTrainerInterface(device = device)
        for name, dis, w, lr in [
            ("sync_dis", sync_dis, 0.8, 0.00001),
            ("seq_dis", seq_dis, 0.2, 0.00001),
            ("frame_dis", frame_dis, 1.0, 0.0001),
        ]:
            if name in pretrained_model_paths:
                path = pretrained_model_paths[name]
                dis.load_state_dict(torch.load(path))
            dis_module = GansModule(
                model = dis,
                optim = optim.Adam(dis.parameters(), lr = lr),
                loss_function = WeightedBCELoss(weight = w, device = device),
            )
            self.__trainer.inject_discriminator(name, dis_module)

        self.__trainer.inject_other_loss_function("l1_loss", VideoL1Loss(weight = 600.0, device = device))
        self.__device = device
        
    def __create_dataloader(self, rootpath, batchsize):
        data_paths = list()
        for path, _ , files in os.walk(rootpath):
            for name in files:
                ext = name.split('.')[-1]
                if ext != 'pkl':
                    continue
                data_paths.append(os.path.join(path, name))

        def data_processing(fd):
            data = pickle.load(fd)
            video = data['video']
            video = torch.tensor(video).float()
            video = normalize(video, [128, 128, 128], [128, 128, 128])
            video = video.transpose(0, 1)
            video_frames = video.shape[1]
            if video_frames > 75:
                video = video[:,:75,:,:]
            elif video_frames < 75:
                more_frames = 75 - video_frames
                dup_frame = torch.unsqueeze(video[:,-1,:,:], dim = 1)
                frames = dup_frame.repeat(1, more_frames, 1, 1)
                video = torch.cat([video, frames], dim = 1)

            audio = data['audio'].astype(np.float)
            audio = audio/audio.max()
            audio = np.pad(audio, (486, 486), 'constant', constant_values = (0, 0))
            audio = torch.tensor(np.expand_dims(audio, axis = 0)).float()

            return (video, audio)

        dataset = PathDataset(data_paths, data_processing)
        params = {
            'batch_size': batchsize,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': True,
        }
        return DataLoader(dataset, **params)

    def __produce_train_data(self):
        for image, audio in self.__train_dataloader:
            image = image.to(self.__device)
            audio = audio.to(self.__device)
            yield (image, audio)

    def __produce_test_data(self):
        for image, audio in self.__test_dataloader:
            image = image.to(self.__device)
            audio = audio.to(self.__device)
            yield (image, audio)

    def __save_model(self, epoch, generator, discriminator_map):
        log.info(f"saving model for epoch {epoch}")
        models_folder = "models"
        epoch_folder = "epoch_{}".format(epoch)
        folder_path = os.path.join(self.__output_path, models_folder, epoch_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        generator_path = os.path.join(folder_path, "generator.pt")
        torch.save(generator.state_dict(), generator_path)

        for k, v in discriminator_map.items():
            dis_path = os.path.join(folder_path, k + '.pt')
            torch.save(v.state_dict(), dis_path)

    def __save_evaluation_data(self, epoch, sample, metrics, orig_data, generated_data):
        if sample != 0:
            return
        data_folder = "data"
        folder_path = os.path.join(self.__output_path, data_folder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        orig_video, audio = orig_data
        generated_data = generated_data.detach().cpu().numpy()
        generated_data = 125.0 * generated_data + 125.0
        generated_data = generated_data.astype(np.uint8)
        orig_video = orig_video.detach().cpu().numpy()
        orig_video = 125.0 * orig_video + 125.0
        orig_video = orig_video.astype(np.uint8)
        audio = (audio.detach().cpu().numpy() * (2 ** 15)).astype(np.int16)
        data = {
            'orig_video': orig_video,
            'audio': audio,
            'generated_data': generated_data,
        }
        data_path = os.path.join(folder_path, "data_{}.pkl".format(epoch))
        with open(data_path, 'wb') as fd:
            pickle.dump(data, fd)

    def start(self):
        self.__trainer.train()

def get_config():
    config = dict()

    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = os.getenv('FIG_DEVICE')
    device = device if device is not None else torch_device
    config['device'] = device

    default_train_datapath = "/media/tuantran/rapid-data/dataset/GRID/face_images"
    train_datapath = os.getenv('FIG_TRAIN_DATAPATH')
    train_datapath = train_datapath if train_datapath is not None else default_train_datapath
    config['train_datapath'] = train_datapath

    default_test_datapath = "./grid-dataset/samples/"
    test_datapath = os.getenv('FIG_TEST_DATAPATH')
    test_datapath = test_datapath if test_datapath is not None else default_test_datapath
    config['test_datapath'] = test_datapath

    default_output_path = "./trainer_output/"
    output_path = os.getenv('FIG_OUTPUT_PATH')
    output_path = output_path if output_path is not None else default_output_path
    config['output_path'] = output_path

    default_pretrained_model_path = "./trainer_output/models/final"
    pretrained_model_path = os.getenv('FIG_PRETRAINED_MODEL_PATH')
    pretrained_model_path = pretrained_model_path if pretrained_model_path is not None else default_pretrained_model_path
    pretrained_model_paths = {}
    if os.path.exists(pretrained_model_path):
        lst = set(os.listdir(pretrained_model_path))
        for name in ['generator', 'sync_dis', 'frame_dis', 'seq_dis']:
            filename = name + '.pt'
            if filename not in lst:
                break
            pretrained_model_paths[name] = os.path.join(pretrained_model_path, filename)
    if len(pretrained_model_paths) == 4:
        config['pretrained_model_paths'] = pretrained_model_paths

    default_epochs = 10
    epochs = os.getenv('FIG_EPOCHS')
    epochs = int(epochs) if epochs is not None else default_epochs
    config['epochs'] = epochs

    default_batchsize = 2
    batchsize = os.getenv('FIG_BATCHSIZE')
    batchsize = int(batchsize) if batchsize is not None else default_batchsize
    config['batchsize'] = batchsize

    return config

def main():
    config = get_config()
    log.info("Running with config: {}".format(config))
    runner = Runner(**config)
    runner.start()

if __name__ == "__main__":
    main()
