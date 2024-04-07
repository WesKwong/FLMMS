import tools.globvar as glob
logger = glob.get('logger')
results_path = glob.get('results_path')
# --------------------------- - -------------------------- #
import time

import torch

from tools.cuda_utils import get_device
import tools.communicator as comm
import datasets
from nodes.models import get_server_model
from configs.hp_prep_tool import hp_preprocess

device = get_device()

def run_server(expt_group):
    for expt_cnt, expt in enumerate(expt_group):
        logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
        hp = hp_preprocess(expt.hyperparameters)
        expt.update_hp(hp)
        expt.log_hp()
        client_ids = range(1, hp['num_client']+1)

        # Load dataset
        dataset = getattr(datasets, hp['dataset'])()
        train_loader = dataset.get_train_loader(hp['batchsize'])
        test_loader = dataset.get_test_loader(hp['batchsize'])

        # scatter dataset to clients
        logger.info("Scattering dataset to clients")
        client_train_loaders = [
            dataset.get_splited_train_loader(hp['batchsize'], client_id)
            for client_id in client_ids
        ]
        comm.scatter(client_train_loaders, client_ids)
        logger.info("Dataset scattered")

        # Init server model
        client_weights = dataset.get_client_weights()
        model_obj = get_server_model(hp)
        server = model_obj(hp, expt, test_loader, client_weights)

        # Start distributed training
        logger.info("Start Distributed Training")
        log_data = dict()
        # init weight with clients
        logger.info("Broadcasting initial weight to clients")
        weight = server.get_weight()
        comm.broadcast(weight, client_ids)
        logger.info("Initial weight broadcasted")
        start_time = time.time()
        for round in range(1, hp["num_rounds"] + 1):
            logger.info(f"Round {round}/{hp['num_rounds']}")
            log_time = time.time() - start_time
            # gather client weight updates
            clients_params = comm.gather(client_ids)
            # aggregate weight updates
            server.aggregate_weight_updates(clients_params, hp['aggregation'])
            # update server weight
            server.update_weight()
            # broadcast aggregated weight updates to clients
            weight_update = server.get_weight_update()
            comm.broadcast(weight_update, client_ids)
            # log
            if not expt.is_log_round(round):
                continue
            client_logs = comm.gather(client_ids)
            log_data[round] = ({
                "weight": server.get_weight(),
                "client_logs": client_logs,
                "log_time": log_time
            })

        # Evaluate
        for round in range(1, hp["num_rounds"] + 1):
            if not expt.is_log_round(round):
                continue
            data = log_data[round]
            server.set_weight(data["weight"])
            logger.info(f"Evaluating...")
            results_trainset_eval = server.evaluate(loader=train_loader, max_samples=5000, verbose=False)
            results_testset_eval = server.evaluate(loader=test_loader, max_samples=10000, verbose=False)
            # log clients
            client_logs = data["client_logs"]
            client_train_losses = [log["train_loss"] for log in client_logs]
            client_lrs = [log["lr"] for log in client_logs]
            client_epochs = [log["epoch"] for log in client_logs]
            client_iters = [log["iteration"] for log in client_logs]
            expt.log({
                f"client_{i+1}_train_loss": loss for i, loss in enumerate(client_train_losses)
            }, printout=True)
            expt.log({
                f"client_{i+1}_lr": lr for i, lr in enumerate(client_lrs)
            }, printout=True)
            expt.log({
                f"client_{i+1}_epoch": epoch for i, epoch in enumerate(client_epochs)
            }, printout=True)
            expt.log({
                f"client_{i+1}_iteration": iteration for i, iteration in enumerate(client_iters)
            }, printout=True)
            # log server
            expt.log({
                "comm_round": round
            })
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
        del server, dataset, train_loader, test_loader, client_train_loaders
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
