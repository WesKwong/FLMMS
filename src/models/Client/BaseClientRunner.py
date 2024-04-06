import tools.GlobVarManager as glob
logger = glob.get('logger')
results_path = glob.get('results_path')
# --------------------------- - -------------------------- #
import os

import torch

from tools.CudaTool import get_device
import tools.CommTool as comm
from models.Client.BaseClientModel import BaseClientModel
from configs.ParamPreprocessor import hp_preprocess

device = get_device()

def run_client(expt_group):
    for expt_cnt, expt in enumerate(expt_group):
        logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
        hp = hp_preprocess(expt.hyperparameters)
        expt.update_hp(hp)
        expt.log_hp()
        server_rank = int(os.environ["WORLD_SIZE"])-1

        # Receive dataset from server
        train_loader = comm.recv(server_rank)

        # Init client model
        client = BaseClientModel(hp, expt, train_loader, int(os.environ["RANK"]))

        # Start distributed training
        logger.info("Start Distributed Training")
        # init weight with server
        weight = comm.recv(server_rank)
        client.set_weight(weight)
        for round in range(1, hp["num_rounds"] + 1):
            # compute weight update
            client.compute_weight_update(hp["local_iters"])
            # send weight update to server
            weight_update = client.get_weight_update()
            comm.send(weight_update, server_rank)
            # recv aggregated weight update from server
            weight_update = comm.recv(server_rank)
            client.set_weight_update(weight_update)
            # sync model
            client.sync_model()
            # log
            client_log = {
                "epoch": client.epoch,
                "iteration": client.iteration,
                "train_loss": client.train_loss,
                "lr": client.current_lr,
            }
            comm.send(client_log, server_rank)

        del client, train_loader
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()