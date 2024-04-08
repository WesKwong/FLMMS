import os
import torch
from .datatool import split_data


class DatasetSplitter(object):
    name = None
    train_set = None
    test_set = None
    split_train_set = dict()

    def __init__(self, path, num_client, data_distribution):
        raw_path = os.path.join(path, self.name, "raw")
        split_path = os.path.join(path, self.name, "split")
        self.load_raw_dataset(raw_path)
        self.split_train_data(data_distribution, num_client, self.train_set)
        self.save_split_dataset(split_path, num_client)

    def load_raw_dataset(self, path):
        raise NotImplementedError

    def split_train_data(self, data_distribution, num_client, train_set):
        split_train_set = split_data(data_distribution, num_client, train_set)
        for i in range(num_client):
            self.split_train_set[i + 1] = split_train_set[i]

    def save_split_dataset(self, path, num_client):
        server_data_dict = {
            "train_set": self.train_set,
            "test_set": self.test_set,
            "split_train_set": self.split_train_set
        }
        client_data_dict_list = [{
            "train_set": self.split_train_set[i + 1]
        } for i in range(num_client)]
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(server_data_dict, os.path.join(path, "server_data.pt"))
        for i in range(num_client):
            torch.save(client_data_dict_list[i],
                       os.path.join(path, f"client_{i + 1}_data.pt"))
