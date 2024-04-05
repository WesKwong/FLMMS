import torch
import torch.nn as nn

import tools.TensorTool as tt
from nnets.NNetManager import get_nn
from tools.CudaTool import get_device
device = get_device()

class BaseModel(object):
    def __init__(self, hp, expt) -> None:
        self.hp = hp
        self.expt = expt

        # get neural network
        self.net = get_nn(hp)

        # parameters
        self.W = {name: value for name, value in self.net.named_parameters()}
        self.dW = {
            name: torch.zeros(value.shape).to(device)
            for name, value in self.W.items()
        }

        # get loss function
        self.loss_fn = self.get_loss_fn(hp)

    def get_loss_fn(self, hp):
        if hp["dataset"] == "CIFAR10":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid dataset: {hp['dataset']}")

    def set_weight(self, W):
        tt.copy(target=self.W, source=W)

    def get_weight(self):
        W = {
            name: torch.zeros(value.shape).to(device)
            for name, value in self.W.items()
        }
        tt.copy(target=W, source=self.W)
        return W

    def set_weight_update(self, dW):
        tt.copy(target=self.dW, source=dW)

    def get_weight_update(self):
        dW = {
            name: torch.zeros(value.shape).to(device)
            for name, value in self.W.items()
        }
        tt.copy(target=dW, source=self.dW)
        return dW