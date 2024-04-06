def get_dataset(hp: dict):
    dataset = hp["dataset"]
    if dataset == 'CIFAR10':
        from datasets.CIFAR10 import get_dataset
        return get_dataset()
    else:
        raise ValueError(f"Invalid dataset: {dataset}")