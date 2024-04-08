from loguru import logger
import tools.globvar as glob

results_path = glob.get('results_path')
# -------------------------------------------------------- #
import time

import torch

import datasets
import tools.communicator as comm
from tools.cuda_utils import get_device
from tools.expt_utils import log_progress
from nodes.models import get_server_model
from configs.hp_prep_tool import hp_preprocess
from configs.config import global_config as config

device = get_device()


def run_server(expt_group):
    for expt_cnt, expt in enumerate(expt_group):
        logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
        # -------------- Prepare Environment ------------- #
        hp = hp_preprocess(expt.hyperparameters)
        expt.update_hp(hp)
        expt.log_hp()
        self_id = 0
        client_ids = range(1, hp['num_client'] + 1)

        # ----------------- Load dataset ----------------- #
        dataset = getattr(datasets, hp['dataset'])(config.data_path, hp['net'],
                                                   self_id)
        train_loader = dataset.get_train_loader(self_id, hp['batchsize'])
        test_loader = dataset.get_test_loader(hp['batchsize'])
        client_weights = dataset.get_client_weights()
        del dataset

        # --------------- Init server model -------------- #
        model_obj = get_server_model(hp)
        server = model_obj(hp, expt, test_loader, client_weights)

        # ================================================ #
        #            Start Distributed Training            #
        # ================================================ #
        logger.info("Start Distributed Training")
        log_data = dict()
        # ----------- init weight with clients ----------- #
        logger.info("Broadcasting initial weight to clients")
        weight = server.get_weight()
        comm.broadcast(weight, client_ids)
        # --------------------- Train -------------------- #
        logger.info("Training...")
        start_time = time.time()
        for round in range(1, hp["num_rounds"] + 1):
            # gather client weight updates
            clients_params = comm.gather(client_ids)
            # aggregate weight updates
            server.aggregate_weight_updates(clients_params, hp['aggregation'])
            # update server weight
            server.update_weight()
            # broadcast aggregated weight updates to clients
            weight_update = server.get_weight_update()
            comm.broadcast(weight_update, client_ids)
            # -------------------------------------------- #
            log_time = time.time() - start_time
            log_progress(start_time, round, hp["num_rounds"])
            # log
            if not expt.is_log_round(round):
                continue
            client_logs = comm.gather(client_ids)
            log_data[round] = ({
                "weight": server.get_weight(),
                "client_logs": client_logs,
                "log_time": log_time
            })

        # ================================================ #
        #                     Evaluate                     #
        # ================================================ #
        for round in range(1, hp["num_rounds"] + 1):
            if not expt.is_log_round(round):
                continue
            data = log_data[round]
            server.set_weight(data["weight"])
            logger.info(f"Evaluating...")
            results_trainset_eval = server.evaluate(loader=train_loader,
                                                    max_samples=5000,
                                                    verbose=False)
            results_testset_eval = server.evaluate(loader=test_loader,
                                                   max_samples=10000,
                                                   verbose=False)
            # log clients
            client_logs = data["client_logs"]
            client_train_losses = [log["train_loss"] for log in client_logs]
            client_lrs = [log["lr"] for log in client_logs]
            client_epochs = [log["epoch"] for log in client_logs]
            client_iters = [log["iteration"] for log in client_logs]
            expt.log(
                {
                    f"client_{i+1}_train_loss": loss
                    for i, loss in enumerate(client_train_losses)
                },
                printout=True)
            expt.log(
                {
                    f"client_{i+1}_lr": lr
                    for i, lr in enumerate(client_lrs)
                },
                printout=True)
            expt.log(
                {
                    f"client_{i+1}_epoch": epoch
                    for i, epoch in enumerate(client_epochs)
                },
                printout=True)
            expt.log(
                {
                    f"client_{i+1}_iteration": iteration
                    for i, iteration in enumerate(client_iters)
                },
                printout=True)
            # log server
            expt.log({"comm_round": round})
            expt.log({
                "train_" + key: value
                for key, value in results_trainset_eval.items()
            })
            expt.log({
                "test_" + key: value
                for key, value in results_testset_eval.items()
            })
            expt.log({"time": log_data[round]["log_time"]})
            expt.save_to_disc(results_path)
        del server, train_loader, test_loader
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
