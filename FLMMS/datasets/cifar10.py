import numpy as np

import torchvision
import torchvision.transforms as transforms

from .basedataset import BaseDataset
from .datasetsplitter import DatasetSplitter


class CIFAR10(BaseDataset):

    def __init__(self, path, id) -> None:
        self.name = "CIFAR10"
        self.n_labels = 10
        super().__init__(path, id)

    def load_data_transform(self):
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


class CIFAR10Splitter(DatasetSplitter):

    def __init__(self, path, num_client, data_distribution):
        self.name = "CIFAR10"
        super().__init__(path, num_client, data_distribution)

    def load_raw_dataset(self, path):
        train_set = torchvision.datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transforms.ToTensor())
        train_set.data = train_set.data.transpose((0, 3, 1, 2))
        test_set.data = test_set.data.transpose((0, 3, 1, 2))
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
        self.train_set = train_set
        self.test_set = test_set
