import numpy as np

import torchvision
import torchvision.transforms as transforms

from .basedataset import BaseDataset
from .datasetsplitter import DatasetSplitter


class CIFAR10(BaseDataset):

    def __init__(self, path, net, id) -> None:
        self.name = "CIFAR10"
        self.n_labels = 10
        super().__init__(path, net, id)

    def load_data_transform(self, net):
        if net == "LeNet5":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        elif net == "AlexNet":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Invalid net: {net}")


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
