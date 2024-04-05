def get_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        from datasets.CIFAR10 import get_dataset
        return get_dataset()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

class BaseDataset(object):
    train_set = None
    test_set = None
    train_loader = None
    test_loader = None

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    def get_train_loader(self):
        raise NotImplementedError

    def get_test_loader(self):
        raise NotImplementedError
