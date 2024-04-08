from loguru import logger
import tools.globvar as glob

results_path = glob.get('results_path')
# -------------------------------------------------------- #
import os

import torch

import datasets
import tools.communicator as comm
from tools.cuda_utils import get_device
from nodes.models import get_client_model
from configs.hp_prep_tool import hp_preprocess
from configs.config import global_config as config

device = get_device()


def run_client(expt_group):
    for expt_cnt, expt in enumerate(expt_group):
        logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
        hp = hp_preprocess(expt.hyperparameters)
        expt.update_hp(hp)
        expt.log_hp()
        server_id = 0
        self_id = int(os.environ["RANK"])

        # Load dataset
        dataset = getattr(datasets, hp["dataset"])(config.data_path, self_id)
        train_loader = dataset.get_train_loader(self_id, hp["batchsize"])
        del dataset

        # Init client model
        model_obj = get_client_model(hp)
        client = model_obj(hp, expt, train_loader, self_id)

        # Start distributed training
        logger.info("Start Distributed Training")
        # init weight with server
        weight = comm.recv(server_id)
        client.set_weight(weight)
        for round in range(1, hp["num_rounds"] + 1):
            # compute weight update
            client.compute_weight_update(hp["local_iters"])
            # send weight update to server
            weight_update = {"dW": client.get_weight_update(), "id": client.id}
            comm.send(weight_update, server_id)
            # recv aggregated weight update from server
            weight_update = comm.recv(server_id)
            client.set_weight_update(weight_update)
            # sync model
            client.sync_model()
            # log
            if not expt.is_log_round(round):
                continue
            client_log = {
                "epoch": client.epoch,
                "iteration": client.iteration,
                "train_loss": client.train_loss,
                "lr": client.current_lr,
            }
            comm.send(client_log, server_id)

        del client, train_loader
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
