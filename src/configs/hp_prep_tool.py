from .config import *

def get_algo_hp(algo):
    name = algo["name"]
    param = algo["param"]
    if name == "None":
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

def hp_preprocess(hp):
    hp.update(model_config.configs_dict)
    hp.update(get_algo_hp(hp["algo"]))
    hp["num_client"] = global_config.num_client
    hp["data_distribution"] = global_config.data_distribution
    hp["num_rounds"] = hp["iteration"] // hp["local_iters"]
    if hp["log_freq"] <= 1:
        raise ValueError("log_freq should be greater than 0")
    hp["log_freq"] = min(int(hp["log_freq"]), hp["num_rounds"])
    return hp