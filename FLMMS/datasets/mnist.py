import numpy as np

import torchvision
import torchvision.transforms as transforms

from .basedataset import BaseDataset
from .datasetsplitter import DatasetSplitter


class MNIST(BaseDataset):

    def __init__(self, path, id) -> None:
        self.name = "MNIST"
        self.n_labels = 10
        super().__init__(path, id)

    def load_data_transform(self):
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078, ), (0.1957, ))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078, ), (0.1957, ))
        ])


class MNISTSplitter(DatasetSplitter):

    def __init__(self, path, num_client, data_distribution):
        self.name = "MNIST"
        super().__init__(path, num_client, data_distribution)

    def load_raw_dataset(self, path):
        train_set = torchvision.datasets.MNIST(root=path,
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())
        test_set = torchvision.datasets.MNIST(root=path,
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())
        train_set.data = train_set.data.numpy().reshape(-1, 1, 28, 28) / 255
        test_set.data = test_set.data.numpy().reshape(-1, 1, 28, 28) / 255
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)
        self.train_set = train_set
        self.test_set = test_set
