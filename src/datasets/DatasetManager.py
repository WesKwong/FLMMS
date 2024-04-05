def get_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        from datasets.CIFAR10 import get_dataset
        return get_dataset()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")