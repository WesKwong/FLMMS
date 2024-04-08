from .lenet5 import LeNet5
from .alexnet import AlexNet


def get_nn(hp):
    net_name = hp["net"]
    dataset = hp["dataset"]
    if net_name == "LeNet5":
        if dataset == "CIFAR10":
            net = LeNet5(10, 3)
        elif dataset == "MNIST":
            net = LeNet5(10, 1)
        elif dataset == "FashionMNIST":
            net = LeNet5(10, 1)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    elif net_name == "AlexNet":
        if dataset == "CIFAR10":
            net = AlexNet(10, 3)
        elif dataset == "MNIST":
            net = AlexNet(10, 1)
        elif dataset == "FashionMNIST":
            net = AlexNet(10, 1)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid net: {net_name}")

    return net
