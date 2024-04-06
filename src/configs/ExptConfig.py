# ------------------ experiment configs ----------------- #
# name: name of the experiment
# dataset: dataset
# net: neural network
# optimizer: optimizer to be used
# scheduler: learning rate scheduler
# lr: learning rate
# min_lr: minimum learning rate
# iteration: number of iterations
# batchsize: batch size
# algo: algorithm
# log_freq: frequency of logging
# -------------------------------------------------------- #
expt_groups = [{
    "name": ["main"],
    "dataset": ["CIFAR10"],
    "net": ["LeNet5"],
    "iteration": [100],
    "algo": [{
        "name": "FedAvg",
        "param": {
            "K": 5
        }
    }],
    "log_freq": [10],
}]

model_hp = {
    "optimizer": "Adam",
    "scheduler": {
        "name": "StepLR",
        "param": {
            "step_size": 1,
            "gamma": 0.5
        }
    },
    "lr": 0.01,
    "min_lr": 0.0001,
    "batchsize": 64
}

def get_algo_hp(algo):
    name = algo["name"]
    param = algo["param"]
    if name == "none":
        return {
            "aggregation": "mean",
            "local_iters": 1
        }
    elif name == "FedAvg":
        return {
            "aggregation": "weighted_mean",
            "local_iters": param["K"],
        }
    else:
        raise ValueError(f"Unknown algorithm: {name}")

