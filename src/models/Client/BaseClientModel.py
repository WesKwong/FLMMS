import torch

import tools.TensorTool as tt
from tools.CudaTool import get_device

device = get_device()

from models.BaseModel import BaseModel


class BaseClientModel(BaseModel):

    def __init__(self, hp, expt, train_loader, id):
        super().__init__(hp, expt)

        self.train_loader = train_loader
        self.iter_train_loader = iter(train_loader)
        self.id = id

        self.old_W = {
            name: torch.zeros(value.shape).to(device)
            for name, value in self.W.items()
        }

        self.min_lr = hp['min_lr']

        # log
        self.epoch = 0
        self.iteration = 0
        self.train_loss = 0.0
        self.schedule_flag = False
        self.current_lr = self.hp['lr']
        self.get_optim()

    def __str__(self) -> str:
        str = f"\nNet: {self.hp['net']}\n"
        str += self.net.__str__()
        str += f"\n\nOptimizer:\n"
        str += f"{self.optimizer}\n"
        str += f"\nLoss Function:\n"
        str += f"{self.loss_fn}\n"
        return str

    def __repr__(self) -> str:
        return self.__str__()

    def get_optim(self):
        # optimizer
        optimizer = getattr(torch.optim, self.hp['optimizer'])
        self.optimizer = optimizer(self.net.parameters(), lr=self.hp['lr'])
        # learning rate scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.hp['scheduler']['name'])
        self.scheduler = scheduler(self.optimizer, **self.hp['scheduler']['param'])

    def sync_model(self):
        tt.add_(target=self.W, source1=self.old_W, source2=self.dW)

    def compute_weight_update(self, iteration):
        # save old weights
        tt.copy(target=self.old_W, source=self.W)

        # train
        self.train_loss = self.train(iteration)

        # get weight update
        tt.sub_(target=self.dW, minuend=self.W, subtrahend=self.old_W)

    def train(self, iteration):
        train_loss = 0.0
        self.net.train()
        for i in range(iteration):
            # get data
            data, label = next(self.iter_train_loader, (None, None))
            if data is None:
                self.epoch += 1
                self.schedule_flag = True
                self.validate()
                self.iter_train_loader = iter(self.train_loader)
                data, label = next(self.iter_train_loader)

            # move to device
            self.net = self.net.to(device)
            data, label = data.to(device), label.to(device)

            # forward pass
            self.optimizer.zero_grad()
            pred = self.net(data)
            loss = self.loss_fn(pred, label)

            # backward pass
            loss.backward()
            self.optimizer.step()

            # log loss
            train_loss += loss.item()
            self.iteration += 1

            # learning rate scheduler
            if self.schedule_flag and self.current_lr > self.min_lr:
                self.scheduler.step()
                self.current_lr = self.optimizer.param_groups[0]['lr']
                if self.current_lr <= self.min_lr:
                    self.optimizer.param_groups[0]['lr'] = self.min_lr
                    self.current_lr = self.min_lr
                self.schedule_flag = False

        train_loss = train_loss / iteration

        return train_loss