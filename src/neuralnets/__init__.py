from .lenet5 import LeNet5

def get_nn(hp):
    net = hp["net"]
    dataset = hp["dataset"]
    if net == "LeNet5":
        if dataset == "CIFAR10":
            return LeNet5(10, 3)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid net: {net}")