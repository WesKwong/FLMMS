import nnets

from nnets.LeNet5 import LeNet5NetGetter

def get_nn(hp):
    if hp['net'] == "LeNet5":
        return LeNet5NetGetter(hp)
    else:
        raise ValueError(f"Invalid net getting: {hp['net']}")