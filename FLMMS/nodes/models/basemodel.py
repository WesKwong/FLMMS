import torch.nn as nn

from tools.tensor_tool import *
from neuralnets import get_nn
from tools.cuda_utils import get_device

device = get_device()


def get_loss_fn(hp):
    if hp["dataset"] == "CIFAR10":
        return nn.CrossEntropyLoss()
    elif hp["dataset"] == "MNIST":
        return nn.CrossEntropyLoss()
    elif hp["dataset"] == "FashionMNIST":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid dataset: {hp['dataset']}")


class BaseModel(object):

    def __init__(self, hp, expt) -> None:
        self.hp = hp
        self.expt = expt

        # get neural network
        self.net = get_nn(hp).to(device)

        # parameters
        self.W = {name: value for name, value in self.net.named_parameters()}
        self.dW = duplicate_zeros_like(self.W, device)

        # get loss function
        self.loss_fn = get_loss_fn(hp)

    def set_weight(self, W):
        assign(target=self.W, source=W)

    def get_weight(self):
        return duplicate(self.W, device)

    def set_weight_update(self, dW):
        assign(target=self.dW, source=dW)

    def get_weight_update(self):
        return duplicate(self.dW, device)
