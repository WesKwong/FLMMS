def get_nn(hp):
    if hp['net'] == 'LeNet5':
        from nnets.LeNet5 import get_net
        return get_net(hp["dataset"])
    else:
        raise ValueError(f"Invalid net: {hp['net']}")