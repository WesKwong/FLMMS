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


class BaseRunner(object):

    def __init__(self, expt) -> None:
        self.expt = expt
        self.prepare_environment()
        self.load_dataset()
        self.init_client_model()
        self.distribute_training()
        self.evaluate()

    def prepare_environment(self):
        self.hp = hp_preprocess(self.expt.hyperparameters)
        self.expt.update_hp(self.hp)
        self.expt.log_hp()
        self.server_id = 0
        self.self_id = int(os.environ["RANK"])

    def load_dataset(self):
        dataset = getattr(datasets,
                          self.hp["dataset"])(config.data_path, self.hp['net'],
                                              self.self_id)
        self.train_loader = dataset.get_train_loader(self.self_id,
                                                     self.hp["batchsize"])
        del dataset

    def init_client_model(self):
        model_obj = get_client_model(self.hp)
        self.client = model_obj(self.hp, self.expt, self.train_loader,
                                self.self_id)

    def distribute_training(self):
        logger.info("Start Distributed Training")
        # -------------- Init weight with server ------------- #
        weight = comm.recv(self.server_id)
        self.client.set_weight(weight)
        # ----------------------- Train ---------------------- #
        logger.info("Training...")
        start_time = time.time()
        log_progress_header(self.hp["num_rounds"])
        for round in range(1, self.hp["num_rounds"] + 1):
            # ------------- compute weight update ------------ #
            logger.debug(f"Computing weight update...")
            compute_time = time.time()
            self.client.compute_weight_update(self.hp["local_iters"])
            logger.debug(f"Compute time: {time.time() - compute_time}")
            # --------- send weight update to server --------- #
            logger.debug(f"Sending weight update to server...")
            send_time = time.time()
            weight_update = {
                "dW": self.client.get_weight_update(),
                "id": self.client.id
            }
            comm.send(weight_update, self.server_id)
            logger.debug(f"Send time: {time.time() - send_time}")
            # --- recv aggregated weight update from server -- #
            logger.debug(f"Receiving aggregated weight update from server...")
            recv_time = time.time()
            weight_update = comm.recv(self.server_id)
            self.client.set_weight_update(weight_update)
            logger.debug(f"Recv time: {time.time() - recv_time}")
            # ------------------ sync model ------------------ #
            logger.debug(f"Syncing model...")
            sync_time = time.time()
            self.client.sync_model()
            logger.debug(f"Sync time: {time.time() - sync_time}")
            # ---------------------- log --------------------- #
            log_progress_detail(start_time, round, self.hp["num_rounds"])
            if self.expt.is_log_round(round):
                client_log = {
                    "epoch": self.client.epoch,
                    "iteration": self.client.iteration,
                    "train_loss": self.client.train_loss,
                    "lr": self.client.current_lr,
                }
                comm.send(client_log, self.server_id)

        del self.client, self.train_loader
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
