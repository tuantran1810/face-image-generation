import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import loguru
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass

@dataclass
class GansModule:
    model: nn.Module
    optim: optim.Optimizer
    loss_function: nn.Module

class GansTrainer:
    def __init__(
        self,
        epochs,
        discriminator_generator_training_ratio = 1,
        log_interval_second = 30,
        logger = loguru.logger,
    ):
        self.__device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__generator_module: Optional[GansModule] = None
        self.__discriminator_modules: dict[str, GansModule] = dict()

        self.__train_dataloader: Optional[DataLoader] = None
        self.__test_dataloader: Optional[DataLoader] = None
        self.__other_loss_functions: dict[str, nn.Module] = dict()

        self.__epochs: int = epochs
        self.__log = logger.info
        self.__log_interval_second: int = log_interval_second
        self.__last_log_time = time.time()
        self.__discriminator_generator_training_ratio = discriminator_generator_training_ratio

    def inject_train_dataloader(self, dataloader):
        if dataloader is None:
            raise Exception("inject invalid dataloader")
        self.__train_dataloader = dataloader
        return self

    def inject_test_dataloader(self, dataloader):
        if dataloader is None:
            raise Exception("inject invalid dataloader")
        self.__test_dataloader = dataloader
        return self

    def inject_generator(self, gen):
        if gen is None or gen.model is None:
            raise Exception("inject invalid generator")
        gen.model = gen.model.to(self.__device)
        self.__generator_module = gen
        return self

    def inject_discriminator(self, name, dis):
        dis.model = dis.model.to(self.__device)
        self.__discriminator_modules[name] = dis
        return self

    def inject_other_loss_function(self, name, loss_function):
        if loss_function is None:
            raise Exception("inject invalid loss function")
        self.__other_loss_functions[name] = loss_function
        return self

    def __metric_log(self, epoch, sample, metrics):
        lst = []
        for k, v in metrics.items():
            lst.append("{}: {:.6f}".format(k, v))
        body = ", ".join(lst)
        self.__log(f"[epoch {epoch} --- sample {sample}] {body}")

    def __do_logging(self, epoch, sample, metrics):
        now = time.time()
        if now - self.__last_log_time < self.__log_interval_second:
            return
        self.__metric_log(epoch, sample, metrics)

    def __generate(self, data, with_grad = True):
        if with_grad:
            self.__generator_module.model.train()
            return self.__generator_module.model(data)
        self.__generator_module.model.eval()
        with torch.no_grad():
            return self.__generator_module.model(data)

    def get_device(self):
        return self.__device

    def train(self):
        if self.__generator_module is None:
            raise Exception("No generator have been injected")
        if len(self.__discriminator_modules) == 0:
            raise Exception("No discriminator have been injected")

        generator = self.__generator_module.model
        generator_optim = self.__generator_module.optim
        for epoch in range(self.__epochs):
            self.__log(f"================================================[epoch {epoch}]================================================")
            for i, orig_data in enumerate(self.__train_dataloader()):
                metrics = dict()
                # train generator
                generator_optim.zero_grad()

                generated_data = self.__generate(orig_data)

                all_loss = list()
                for _, discriminator in self.__discriminator_modules.items():
                    dis = discriminator.model
                    xhat = dis(orig_data, generated_data, discriminator_training = False)
                    y = dis.suggested_generator_training_label(xhat).to(self.__device)
                    loss = discriminator.loss_function(xhat, y)
                    all_loss.append(loss)

                for _, loss_function in self.__other_loss_functions.items():
                    loss = loss_function(orig_data, generated_data)
                    all_loss.append(loss)

                generator_loss = sum(all_loss)
                generator_loss.backward()

                generator_optim.step()
                
                metrics["generator_loss"] = generator_loss
                # train for discriminators
                generated_data = generated_data.detach()
                for d_name, discriminator in self.__discriminator_modules.items():
                    discriminator.optim.zero_grad()
                    dis = discriminator.model
                    xhat = dis(orig_data, generated_data, discriminator_training = True)
                    y = dis.suggested_discriminator_training_label(xhat).to(self.__device)
                    loss = discriminator.loss_function(xhat, y)
                    loss.backward()
                    discriminator.optim.step()
                    metrics[d_name + '_loss'] = loss

                self.__do_logging(epoch, i, metrics)
            
            if self.__test_dataloader is None:
                continue

            metrics = dict()
            cnt = 0.0
            with torch.no_grad():
                for orig_data in self.__test_dataloader():
                    cnt += 1.0
                    generated_data = self.__generate(orig_data)

                    all_loss = list()
                    for _, discriminator in self.__discriminator_modules.items():
                        dis = discriminator.model
                        xhat = dis(orig_data, generated_data, discriminator_training = False)
                        y = dis.suggested_generator_training_label(xhat).to(self.__device)
                        loss = discriminator.loss_function(xhat, y)
                        all_loss.append(loss)

                    for _, loss_function in self.__other_loss_functions.items():
                        loss = loss_function(orig_data, generated_data)
                        all_loss.append(loss)

                    generator_loss = sum(all_loss)
                    metrics["generator_loss"] += generator_loss

                    for d_name, discriminator in self.__discriminator_modules.items():
                        dis = discriminator.model
                        xhat = dis(orig_data, generated_data, discriminator_training = False)
                        y = dis.suggested_discriminator_training_label(xhat).to(self.__device)
                        loss = discriminator.loss_function(xhat, y)
                        loss.backward()
                        metrics[d_name + '_loss'] += loss
            
            for k, v in metrics.items():
                metrics[k] = v/cnt

            self.__metric_log(epoch, -1, metrics)
            self.__log("================================================================================================")
