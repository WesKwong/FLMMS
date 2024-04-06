import torch.nn as nn
import torch.nn.functional as F


def LeNet5NetGetter(hp):
    dataset = hp["dataset"]
    if dataset == "CIFAR10":
        return LeNet5(10, 3)
    else:
        raise ValueError(f"Invalid dataset getting LeNet5: {dataset}")


class LeNet5(nn.Module):

    def __init__(self, n_classes, in_channels):
        """
        n_classes: number of classes in the dataset
        in_channels: number of channels in the input data
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
