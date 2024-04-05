import nnets

def get_nn(hp):
    nn_getter = getattr(nnets, hp['net'])
    return nn_getter(hp)