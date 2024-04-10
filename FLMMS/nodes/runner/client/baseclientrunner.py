from loguru import logger
import tools.globvar as glob

results_path = glob.get('results_path')
# -------------------------------------------------------- #
import os
import time

import torch

import datasets
import tools.communicator as comm
from tools.cuda_utils import get_device
from tools.expt_utils import log_progress_header, log_progress_detail
from nodes.models import get_client_model
from configs.hp_prep_tool import hp_preprocess
from configs.config import global_config as config

device = get_device()


def run_client(expt):
    # -------------- Prepare Environment ------------- #
    hp = hp_preprocess(expt.hyperparameters)
    expt.update_hp(hp)
    expt.log_hp()
    server_id = 0
    self_id = int(os.environ["RANK"])

    # ----------------- Load dataset ----------------- #
    dataset = getattr(datasets, hp["dataset"])(config.data_path, hp['net'],
                                               self_id)
    train_loader = dataset.get_train_loader(self_id, hp["batchsize"])
    del dataset

    # --------------- Init client model -------------- #
    model_obj = get_client_model(hp)
    client = model_obj(hp, expt, train_loader, self_id)

    # ================================================ #
    #            Start Distributed Training            #
    # ================================================ #
    logger.info("Start Distributed Training")
    # -------------- Init weight with server ------------- #
    weight = comm.recv(server_id)
    client.set_weight(weight)
    # ----------------------- Train ---------------------- #
    logger.info("Training...")
    start_time = time.time()
    log_progress_header(hp["num_rounds"])
    for round in range(1, hp["num_rounds"] + 1):
        # ------------- compute weight update ------------ #
        logger.debug(f"Computing weight update...")
        compute_time = time.time()
        client.compute_weight_update(hp["local_iters"])
        logger.debug(f"Compute time: {time.time() - compute_time}")
        # --------- send weight update to server --------- #
        logger.debug(f"Sending weight update to server...")
        send_time = time.time()
        weight_update = {"dW": client.get_weight_update(), "id": client.id}
        comm.send(weight_update, server_id)
        logger.debug(f"Send time: {time.time() - send_time}")
        # --- recv aggregated weight update from server -- #
        logger.debug(f"Receiving aggregated weight update from server...")
        recv_time = time.time()
        weight_update = comm.recv(server_id)
        client.set_weight_update(weight_update)
        logger.debug(f"Recv time: {time.time() - recv_time}")
        # ------------------ sync model ------------------ #
        logger.debug(f"Syncing model...")
        sync_time = time.time()
        client.sync_model()
        logger.debug(f"Sync time: {time.time() - sync_time}")
        # ---------------------- log --------------------- #
        log_progress_detail(start_time, round, hp["num_rounds"])
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
