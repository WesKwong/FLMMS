def get_dataset(hp):
    if hp["dataset"] == 'CIFAR10':
        from datasets.CIFAR10 import get_dataset
        return get_dataset()
    else:
        raise ValueError(f"Invalid dataset: {hp["dataset"]}")