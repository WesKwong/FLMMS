from loguru import logger
from configs.config import expt_group_config_manager

from tools.cuda_utils import get_device

device = get_device()

import tools.communicator as comm
from tools.expt_manager import create_experiments
from nodes.runner import *


def run_node(expt):
    if comm.is_server():
        run_server = get_server_runner(expt.hyperparameters)
        run_server(expt)
    else:
        run_client = get_client_runner(expt.hyperparameters)
        run_client(expt)


def main():
    comm.init_communication_group()
    logger.info(f"Experiment Running on {device}...")

    expt_groups_configs = expt_group_config_manager.get_expt_groups_configs()
    expt_groups = create_experiments(expt_groups_configs)
    for i, (name, expt_group) in enumerate(expt_groups.items()):
        logger.info(f"Running ({i+1}/{len(expt_groups)}) group: {name}")
        for expt_cnt, expt in enumerate(expt_group):
            logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
            run_node(expt)

    logger.info("Experiment Done!")
    if comm.is_server():
        comm.destroy_communication_group()
        logger.info("Press <Ctrl-C> to exit monitoring.")
