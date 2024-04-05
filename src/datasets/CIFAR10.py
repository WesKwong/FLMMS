import os

import torch
import torchvision
import torchvision.transforms as transforms

from datasets.DatasetManager import BaseDataset
from configs.MainConfig import config


def get_dataset():
    return CIFAR10()


class CIFAR10(BaseDataset):

    def __init__(self) -> None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        dataset_path = os.path.join(config.data_path, 'CIFAR10')
        train_set = torchvision.datasets.CIFAR10(root=dataset_path,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=dataset_path,
                                                train=False,
                                                download=True,
                                                transform=test_transform)
        self.train_set = train_set
        self.test_set = test_set

    def get_train_loader(self, batch_size):
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.dataloader_workers)

        return train_loader

    def get_test_loader(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.dataloader_workers)

        return test_loader
