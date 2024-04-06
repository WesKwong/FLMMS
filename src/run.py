# ------------- init global variables manager ------------ #
import tools.GlobVarManager as glob
glob._init()
# ---------------------- init config --------------------- #
from configs.MainConfig import config
# ------------------ init logger handler ----------------- #
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, level=config.log_level)
glob.set('logger', logger)
# -------------------- set random seed ------------------- #
from tools.ExptUtils import set_seed
set_seed(config.random_seed)
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


