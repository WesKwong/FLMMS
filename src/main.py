import tools.GlobVarManager as glob
logger = glob.get('logger')
# --------------------------- - -------------------------- #
from configs.MainConfig import config

from tools.CudaTool import get_device
device = get_device()

import tools.CommTool as comm
from models.Server.BaseServerRunner import run_server
from models.Client.BaseClientRunner import run_client
from tools.ExptManager import create_experiments


def run_node(expt_group):
    if comm.is_server():
        run_server(expt_group)
    else:
        run_client(expt_group)


def main():
    logger.info(f"Experiment Running on {device}...")
    comm.init_communication_group()

    expt_groups_configs = config.get_expt_groups_configs()
    expt_groups = create_experiments(expt_groups_configs)
    for i, (name, expt_group) in enumerate(expt_groups.items()):
        logger.info(f"Running ({i+1}/{len(expt_groups)}) group: {name}")
        run_node(expt_group)

    comm.destroy_communication_group()
    logger.info("Experiment Done!")
