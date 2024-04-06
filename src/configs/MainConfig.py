import itertools as it
from configs.ExptConfig import expt_groups

class Config(object):
    # -------------------- global configs ------------------- #
    expt_name = None
    data_path = 'data/'
    results_path = 'results/'
    random_seed = 42
    log_level = 'INFO'
    num_client = 3 # number of clients
    data_distribution = {
        "iid": True,
        "customize": True,
        "cus_distribution": [5,5,5]
    },
    dataloader_workers = 4
    device = 'cpu' # 'cpu' or 'cuda'
    cuda_device = [0, 1, 2, 3] # available when device is 'cuda',
                               # cuda_device=[0, 1, 2] means server uses GPU 0
                               # and client 1 uses GPU 1, client 2 uses GPU 2
    # --------------------------- - -------------------------- #
    def __init__(self):
        self.check_is_global_valid()
        self.expt_groups = expt_groups

    def check_is_global_valid(self):
        if self.num_client < 1:
            raise ValueError("num_client should be greater than 0")
        if self.num_client != len(self.data_distribution["cus_distribution"]):
            raise ValueError("num_client should be equal to the length of cus_distribution")

    def get_expt_groups_configs(self):
        expt_groups_configs = {}
        for expt_group in self.expt_groups:
            combinations = it.product(*(expt_group[name] for name in expt_group))
            expt_group_configs = [{key : value[i] for i,key in enumerate(expt_group)}for value in combinations]
            expt_groups_configs[expt_group['name'][0]] = expt_group_configs
        return expt_groups_configs

config = Config()