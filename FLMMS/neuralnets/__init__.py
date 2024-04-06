from .lenet5 import LeNet5

def get_nn(hp):
    net_name = hp["net"]
    dataset = hp["dataset"]
    if net_name == "LeNet5":
        if dataset == "CIFAR10":
            net = LeNet5(10, 3)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid net: {net_name}")

    return net