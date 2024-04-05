import tools.GlobVarManager as glob
logger = glob.get('logger')
# --------------------------- - -------------------------- #
from configs.MainConfig import config

from tools.CudaTool import get_device
device = get_device()

import tools.CommTool as comm
from models.Server.BaseServerRunner import run_server
from models.Client.BaseClientRunner import run_client


def main():
    logger.info("Experiment Running...")
    comm.init_communication_group()
    if comm.is_server():
        run_server()
    else:
        run_client()
    comm.destroy_communication_group()
    logger.info("Experiment Done!")
