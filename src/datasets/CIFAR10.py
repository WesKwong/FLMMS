import os
import numpy as np

import torchvision
import torchvision.transforms as transforms

from datasets.BaseDataset import BaseDataset
from configs.MainConfig import config


def get_dataset():
    return CIFAR10()


class CIFAR10(BaseDataset):

    def __init__(self) -> None:
        # CIFAR10 dataset
        dataset_path = os.path.join(config.data_path, 'CIFAR10')
        train_set = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=True,
            download=True,
            transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(
            root=dataset_path,
            train=False,
            download=True,
            transform=transforms.ToTensor())
        train_set.data = train_set.data.transpose((0, 3, 1, 2))
        test_set.data = test_set.data.transpose((0, 3, 1, 2))
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
        self.train_set = train_set
        self.test_set = test_set
        self.n_labels = 10
        # CIFAR10 transform
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        # split train data
        self.split_train_data()