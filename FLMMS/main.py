from loguru import logger
from configs.config import expt_group_config_manager

from tools.cuda_utils import get_device

device = get_device()

import tools.communicator as comm
from tools.expt_manager import create_experiments
from nodes.runner import *


def run_node(expt_group):
    if comm.is_server():
        run_server = get_server_runner(expt_group[0].hyperparameters)
        run_server(expt_group)
    else:
        run_client = get_client_runner(expt_group[0].hyperparameters)
        run_client(expt_group)


def main():
    logger.info(f"Experiment Running on {device}...")
    comm.init_communication_group()

    expt_groups_configs = expt_group_config_manager.get_expt_groups_configs()
    expt_groups = create_experiments(expt_groups_configs)
    for i, (name, expt_group) in enumerate(expt_groups.items()):
        logger.info(f"Running ({i+1}/{len(expt_groups)}) group: {name}")
        run_node(expt_group)

    comm.destroy_communication_group()
    logger.info("Experiment Done!")
    if comm.is_server():
        logger.info("Press <Ctrl-C> to exit monitoring.")
