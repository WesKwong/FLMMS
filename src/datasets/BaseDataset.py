import tools.GlobVarManager as glob
logger = glob.get('logger')
# --------------------------- - -------------------------- #
from configs.MainConfig import config

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np

from tools.DataTool import split_data


class BaseDataset(object):
    train_set = None
    test_set = None
    train_transform = None
    test_transform = None
    n_labels = None
    clients_weight = dict()
    splited_train_set = dict()

    def log_split(self, splited_data) -> str:
        sum = 0
        logger.info("Data split:")
        for i, client in enumerate(splited_data):
            split = np.sum(client[1].reshape(1, -1) == np.arange(self.n_labels).reshape(
                -1, 1),
                        axis=1)
            logger.info(" - Client {}: {}, sum = {}".format(i, split, split.sum()))
            sum += split.sum()
        logger.info(f'sum = {sum}')

    def get_clients_weight(self):
        return self.clients_weight

    def get_split_train_loader(self, batch_size, client_id):
        client_train_set = self.split_train_set[client_id]
        data = client_train_set[0]
        labels = client_train_set[1]
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

    def split_train_data(self):
        splited_train_set = split_data(config.data_distribution,
                                            config.num_clients, self.train_set)
        self.log_split(splited_train_set)
        for i, train_set in enumerate(splited_train_set):
            self.split_train_set[i+1] = train_set
            self.clients_weight[i+1] = len(train_set[0])


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
