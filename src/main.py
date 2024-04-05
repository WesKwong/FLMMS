import tools.GlobVarManager as glob
logger = glob.get('logger')
# --------------------------- - -------------------------- #
import torch

from tools.ExptUtils import set_seed
from tools.CudaTool import get_device
from configs.MainConfig import config
set_seed(config.random_seed)

def main():
    set_seed(config.random_seed)
    logger.info(f"Running on {get_device()}")
    tensor = torch.tensor([1, 2, 3]).to(get_device())
    logger.info(f"Tensor: {tensor}")