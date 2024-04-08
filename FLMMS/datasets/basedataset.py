from loguru import logger

import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np


class BaseDataset(object):
    name = None
    n_labels = None
    train_set = None
    test_set = None
    train_transform = None
    test_transform = None
    split_train_set = dict()
    client_weights = dict()

    def __init__(self, path, id) -> None:
        split_data_path = os.path.join(path, self.name, "split")
        # server data
        if id == 0:
            self.load_server_data(split_data_path)
        # client data
        else:
            self.load_client_data(split_data_path, id)
        self.load_data_transform()

    def load_server_data(self, path):
        data_dict = torch.load(os.path.join(path, "server_data.pt"))
        self.train_set = data_dict["train_set"]
        self.test_set = data_dict["test_set"]
        split_train_set = data_dict["split_train_set"]
        self.split_train_set = split_train_set
        for id, train_set in split_train_set.items():
            self.client_weights[id] = len(train_set[0])
        self.log_split(split_train_set)

    def load_client_data(self, path, client_id):
        data_dict = torch.load(
            os.path.join(path, f"client_{client_id}_data.pt"))
        self.train_set = data_dict["train_set"]

    def log_split(self, splited_data) -> str:
        sum = 0
        logger.info("Data split:")
        for id, client in splited_data.items():
            split = np.sum(client[1].reshape(1, -1) == np.arange(
                self.n_labels).reshape(-1, 1),
                           axis=1)
            logger.info(" - Client {}: {}, sum = {}".format(
                id, split, split.sum()))
            sum += split.sum()
        logger.info(f'sum = {sum}')

    def load_data_transform(self):
        raise NotImplementedError

    def get_train_loader(self, id, batch_size):
        if id == 0:
            data = self.train_set.data
            labels = self.train_set.targets
        else:
            data = self.train_set[0]
            labels = self.train_set[1]
        loader = DataLoader(CustomerDataset(data, labels,
                                            self.train_transform),
                            batch_size=batch_size,
                            shuffle=True)
        return loader

    def get_test_loader(self, batch_size):
        data = self.test_set.data
        labels = self.test_set.targets
        loader = DataLoader(CustomerDataset(data, labels, self.test_transform),
                            batch_size=batch_size,
                            shuffle=False)
        return loader

    def get_split_train_loader(self, batch_size, client_id):
        client_train_set = self.split_train_set[client_id]
        data = client_train_set[0]
        labels = client_train_set[1]
        loader = DataLoader(CustomerDataset(data, labels,
                                            self.train_transform),
                            batch_size=batch_size,
                            shuffle=True)
        return loader

    def get_client_weights(self):
        return self.client_weights


class CustomerDataset(Dataset):
    '''
    A custom Dataset class for client
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        if inputs.dtype == 'uint8':
            self.data = torch.tensor(inputs)
        else:
            self.data = torch.tensor(inputs).float()
        self.targets = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.data.shape[0]
