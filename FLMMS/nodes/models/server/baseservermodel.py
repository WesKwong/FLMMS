from loguru import logger
import torch

from tools.tensor_tool import *
from tools.cuda_utils import get_device
device = get_device()

from ..basemodel import BaseModel


class BaseServerModel(BaseModel):

    def __init__(self, hp, expt, test_loader, client_weights):
        super().__init__(hp, expt)
        self.test_loader = test_loader
        client_weights_list = [0]
        client_weights_list.extend([client_weights[i] for i in range(1, len(client_weights)+1)])
        self.client_weights = torch.Tensor(client_weights_list).to(device)

    def update_weight(self):
        add(target=self.W, source=self.dW)

    def aggregate_weight_updates(self, clients_params, aggregation="mean"):
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            mean(target=self.dW,
                       sources=[param["dW"] for param in clients_params])
        elif aggregation == "weighted_mean":
            weighted_mean(
                target=self.dW,
                sources=[param["dW"] for param in clients_params],
                weights=torch.stack([
                    self.client_weights[param["id"]] for param in clients_params
                ]))
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation}")

    def evaluate(self, loader=None, max_samples=None, verbose=True):
        self.net.eval()
        correct = 0
        total = 0
        eval_loss = 0.0
        eval_steps = 0
        if loader is None:
            loader = self.test_loader
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                # correct
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # loss
                eval_loss += self.loss_fn(outputs, labels).item()
                eval_steps += 1

                if max_samples is not None and total >= max_samples:
                    break

            if verbose:
                logger.info(f"Evaluated on {total} samples ({eval_steps} batches)")

            results = {
                "loss": eval_loss / eval_steps,
                "accuracy": correct / total
            }
        return results