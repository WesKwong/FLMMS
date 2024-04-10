from loguru import logger
import tools.globvar as glob

results_path = glob.get('results_path')
# -------------------------------------------------------- #
import time

import torch

import datasets
import tools.communicator as comm
from tools.cuda_utils import get_device
from tools.expt_utils import log_progress_header, log_progress_detail
from nodes.models import get_server_model
from configs.hp_prep_tool import hp_preprocess
from configs.config import global_config as config

device = get_device()


class BaseRunner(object):

    def __init__(self, expt) -> None:
        self.expt = expt
        self.prepare_environment()
        self.load_dataset()
        self.init_server_model()
        self.distribute_training()
        self.evaluate()

    def prepare_environment(self):
        self.hp = hp_preprocess(self.expt.hyperparameters)
        self.expt.update_hp(self.hp)
        self.expt.log_hp()
        self.self_id = 0
        self.client_ids = range(1, self.hp['num_client'] + 1)

    def load_dataset(self):
        dataset = getattr(datasets,
                          self.hp['dataset'])(config.data_path, self.hp['net'],
                                              self.self_id)
        self.train_loader = dataset.get_train_loader(self.self_id,
                                                     self.hp['batchsize'])
        self.test_loader = dataset.get_test_loader(self.hp['batchsize'])
        self.client_weights = dataset.get_client_weights()
        del dataset

    def init_server_model(self):
        model_obj = get_server_model(self.hp)
        self.server = model_obj(self.hp, self.expt, self.test_loader,
                                self.client_weights)

    def distribute_training(self):
        logger.info("Start Distributed Training")
        self.log_data = dict()
        # ----------- init weight with clients ----------- #
        logger.info("Broadcasting initial weight to clients")
        weight = self.server.get_weight()
        comm.broadcast(weight, self.client_ids)
        # --------------------- Train -------------------- #
        logger.info("Training...")
        start_time = time.time()
        log_progress_header(self.hp["num_rounds"])
        for round in range(1, self.hp["num_rounds"] + 1):
            # --------- gather client weight updates --------- #
            logger.debug(f"Gathering client weight updates...")
            gather_time = time.time()
            clients_params = comm.gather(self.client_ids)
            logger.debug(f"Gather time: {time.time() - gather_time}")
            # ----------- aggregate weight updates ----------- #
            logger.debug(f"Aggregating weight updates...")
            aggregate_time = time.time()
            self.server.aggregate_weight_updates(clients_params,
                                                 self.hp['aggregation'])
            logger.debug(f"Aggregate time: {time.time() - aggregate_time}")
            # ------------- update server weight ------------- #
            logger.debug(f"Updating server weight...")
            update_time = time.time()
            self.server.update_weight()
            logger.debug(f"Update time: {time.time() - update_time}")
            # ------ broadcast aggregated weight updates ----- #
            logger.debug(
                f"Broadcasting aggregated weight updates to clients...")
            broadcast_time = time.time()
            weight_update = self.server.get_weight_update()
            comm.broadcast(weight_update, self.client_ids)
            logger.debug(f"Broadcast time: {time.time() - broadcast_time}")
            # ---------------------- log --------------------- #
            log_time = time.time() - start_time
            log_progress_detail(start_time, round, self.hp["num_rounds"])
            if self.expt.is_log_round(round):
                client_logs = comm.gather(self.client_ids)
                self.log_data[round] = ({
                    "weight": self.server.get_weight(),
                    "client_logs": client_logs,
                    "log_time": log_time
                })

    def evaluate(self):
        logger.info("Start Evaluation")
        for round in range(1, self.hp["num_rounds"] + 1):
            if self.expt.is_log_round(round):
                data = self.log_data[round]
                self.server.set_weight(data["weight"])
                logger.info(f"Evaluating...")
                results_trainset_eval = self.server.evaluate(
                    loader=self.train_loader, max_samples=5000, verbose=False)
                results_testset_eval = self.server.evaluate(
                    loader=self.test_loader, max_samples=10000, verbose=False)
                # -------------- log client -------------- #
                client_logs = data["client_logs"]
                client_train_losses = [
                    log["train_loss"] for log in client_logs
                ]
                client_lrs = [log["lr"] for log in client_logs]
                client_epochs = [log["epoch"] for log in client_logs]
                client_iters = [log["iteration"] for log in client_logs]
                self.expt.log(
                    {
                        f"client_{i+1}_train_loss": loss
                        for i, loss in enumerate(client_train_losses)
                    },
                    printout=True)
                self.expt.log(
                    {
                        f"client_{i+1}_lr": lr
                        for i, lr in enumerate(client_lrs)
                    },
                    printout=True)
                self.expt.log(
                    {
                        f"client_{i+1}_epoch": epoch
                        for i, epoch in enumerate(client_epochs)
                    },
                    printout=True)
                self.expt.log(
                    {
                        f"client_{i+1}_iteration": iteration
                        for i, iteration in enumerate(client_iters)
                    },
                    printout=True)
                # -------------- log server -------------- #
                self.expt.log({"comm_round": round})
                self.expt.log({
                    "train_" + key: value
                    for key, value in results_trainset_eval.items()
                })
                self.expt.log({
                    "test_" + key: value
                    for key, value in results_testset_eval.items()
                })
                self.expt.log({"time": self.log_data[round]["log_time"]})
                self.expt.save_to_disc(results_path)

        del self.server, self.train_loader, self.test_loader
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
