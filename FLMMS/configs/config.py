import itertools as it

def get_configs_dict(configs):
    configs_dict = {}
    for config_name, config_value in configs.items():
        if config_name.startswith('__') and config_name.endswith('__'):
            continue
        configs_dict[config_name] = config_value
    return configs_dict

# ======================================================== #
#                 Experiment Group Configs                 #
# ======================================================== #
class ExptGroupConfig1(object):
    group_name = ["main"]
    dataset = ["CIFAR10"]
    net = ["LeNet5"]
    iteration = [100]
    algo = [{"name": "FedAvg", "param": {"K": 5}}]
    log_freq = [5]
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())

class ExptGroupConfig2(object):
    group_name = ["main"]
    dataset = ["CIFAR10"]
    net = ["LeNet5"]
    iteration = [100]
    algo = [{"name": "none", "param": {}}]
    log_freq = [5]
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())

class ExptGroupConfigManager(object):
    expt_groups = [
        ExptGroupConfig1.configs_dict
    ]
    def get_expt_groups_configs(self):
        expt_groups_configs = {}
        for expt_group in self.expt_groups:
            combinations = it.product(*(expt_group[name] for name in expt_group))
            expt_group_configs = [{key : value[i] for i,key in enumerate(expt_group)}for value in combinations]
            expt_groups_configs[expt_group['group_name'][0]] = expt_group_configs
        return expt_groups_configs

# ======================================================== #
#                       Global Config                      #
# ======================================================== #
class GlobalConfig(object):
    # -------------------- environment ------------------- #
    expt_name = None
    data_path = 'data/'
    results_path = 'results/'
    log_level = 'INFO'
    random_seed = 42
    # ---------------------- device ---------------------- #
    dataloader_workers = 4
    device = 'cpu'
    cuda_device = [0, 1, 2, 3] # available when device is 'cuda'
    # cuda_device=[0, 1, 2] means server uses GPU 0
    # and client 1 uses GPU 1, client 2 uses GPU 2
    # ---------------------- client ---------------------- #
    num_client = 3
    data_distribution = {
        "iid": True,
        "customize": True,
        "cus_distribution": [5,5,5]
    }
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())

# ======================================================== #
#                       Model Config                       #
# ======================================================== #
class ModelConfig(object):
    optimizer = "Adam"
    scheduler = {
        "name": "StepLR",
        "param": {
            "step_size": 1,
            "gamma": 0.5
        }
    }
    lr = 0.01
    min_lr = 0.0001
    batchsize = 64
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())

expt_group_config_manager = ExptGroupConfigManager()
global_config = GlobalConfig()
model_config = ModelConfig()