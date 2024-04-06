# ------------- init global variables manager ------------ #
import tools.globvar as glob
glob._init()
# ---------------------- init config --------------------- #
from configs import global_config
# ------------------ init logger handler ----------------- #
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, level=global_config.log_level)
glob.set('logger', logger)
# -------------------- set random seed ------------------- #
from tools.expt_utils import set_seed
set_seed(global_config.random_seed)
# ------------------------- run main ------------------------- #
import os
RESULTS_PATH = os.environ['RESULTS_PATH']
glob.set('results_path', RESULTS_PATH)

@logger.catch
def run():
    from main import main
    main()

if __name__ == "__main__":
    run()


