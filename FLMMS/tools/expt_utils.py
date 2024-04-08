import os
import time
import random
import numpy as np
import torch
from loguru import logger

import datasets

def get_time_str():
    time_str = time.strftime("%Y%m%d%H%M%S")
    year = time_str[2:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hour = time_str[8:10]
    minute = time_str[10:12]
    second = time_str[12:14]
    return year + month + day + hour + minute + second

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_unique_results_path(path, expt_name=None):
    unique_name = get_time_str()
    if expt_name is not None:
        unique_name = unique_name + "_" + expt_name
    unique_results_path = os.path.join(path, unique_name)
    if not os.path.exists(unique_results_path):
        os.makedirs(unique_results_path)
    return unique_results_path

def get_expt_dataset_config_set(expt_groups_configs):
    expt_dataset_config_set = set()
    for group_name, expt_group_configs in expt_groups_configs.items():
        for expt_group_config in expt_group_configs:
            dataset = expt_group_config['dataset']
            expt_dataset_config_set.add(dataset)
    return expt_dataset_config_set

def prepare_datasets(path, num_client, data_distribution, expt_groups_configs):
    expt_dataset_config_set = get_expt_dataset_config_set(expt_groups_configs)
    for dataset in expt_dataset_config_set:
        logger.info(f"Preparing dataset: {dataset}")
        getattr(datasets, dataset + "Splitter")(path, num_client, data_distribution)