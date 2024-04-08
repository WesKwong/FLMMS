import numpy as np

import torchvision
import torchvision.transforms as transforms

from .basedataset import BaseDataset
from .datasetsplitter import DatasetSplitter


class FashionMNIST(BaseDataset):

    def __init__(self, path, net, id) -> None:
        self.name = "FashionMNIST"
        self.n_labels = 10
        super().__init__(path, net, id)

    def load_data_transform(self, net):
        if net == "LeNet5":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        elif net == "AlexNet":
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
        else:
            raise ValueError(f"Invalid net: {net}")


class FashionMNISTSplitter(DatasetSplitter):

    def __init__(self, path, num_client, data_distribution):
        self.name = "FashionMNIST"
        super().__init__(path, num_client, data_distribution)

    def load_raw_dataset(self, path):
        train_set = torchvision.datasets.FashionMNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.ToTensor())
        test_set = torchvision.datasets.FashionMNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.ToTensor())
        train_set.data = train_set.data.numpy().reshape(-1, 1, 28, 28) / 255
        test_set.data = test_set.data.numpy().reshape(-1, 1, 28, 28) / 255
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
        self.train_set = train_set
        self.test_set = test_set
