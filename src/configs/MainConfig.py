import itertools as it

class Config(object):
    # -------------------- global settings ------------------- #
    expt_name = None
    data_path = 'data/'
    results_path = 'results/'
    random_seed = 42
    log_level = 'INFO'
    n_client = 3 # number of clients
    device = 'cpu' # 'cpu' or 'cuda'
    cuda_device = [0, 1, 2, 3] # available when device is 'cuda',
                               # cuda_device=[0, 1, 2] means server uses GPU 0
                               # and client 1 uses GPU 1, client 2 uses GPU 2
    # ------------------ experiment settings ----------------- #
    # name: name of the experiment
    # dataset: dataset
    # net: neural network
    # optimizer: optimizer to be used
    # scheduler: learning rate scheduler
    # lr: learning rate
    # min_lr: minimum learning rate
    # iteration: number of iterations
    # batchsize: batch size
    # algo: algorithm
    # log_freq: frequency of logging
    # --------------------------- - -------------------------- #
    default = {'name': ['main'],
               'dataset': ['CIFAR10'],
               'net': ['CNN'],
               'optimizer': ['Adam'],
               'scheduler': [['StepLR', {'step_size': 1, 'gamma': 0.5}]],
               'lr': [0.01],
               'min_lr': [0.0001],
               'iteration': [100],
               'batchsize': [64],
               'algo': [['FedAvg', {'K': 5}]],
               'log_freq': [10]}
    # --------------------------- - -------------------------- #
    def __init__(self):
        default = self.default
        self.expt_settings = [default]

    def get_expt_settings(self):
        expt_settings = {}
        for setting in self.expt_settings:
            combinations = it.product(*(setting[name] for name in setting))
            expt_setting = [{key : value[i] for i,key in enumerate(setting)}for value in combinations]
            expt_settings[setting['name'][0]] = expt_setting
        return expt_settings

config = Config()