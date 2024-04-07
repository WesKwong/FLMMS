# ======================================================== #
#                    Prepare Environment                   #
# ======================================================== #
import sys
from loguru import logger
import tools.globvar as glob
from configs import global_config as config

glob._init()
logger.remove()
logger.add(sys.stdout, level=config.log_level, backtrace=True, diagnose=True)
# ======================================================== #
#                           Main                           #
# ======================================================== #
import os
from tools.expt_utils import set_seed

RESULTS_PATH = os.environ['RESULTS_PATH']
glob.set('results_path', RESULTS_PATH)
set_seed(config.random_seed)


@logger.catch
def run():
    from main import main
    main()


if __name__ == "__main__":
    run()
