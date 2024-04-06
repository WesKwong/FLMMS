from configs.ExptConfig import model_hp, get_algo_hp

def hp_preprocess(hp):
    hp.update(model_hp)
    hp.update(get_algo_hp(hp["algo"]))
    hp["num_rounds"] = hp["iteration"] // hp["local_iters"]
    if hp["log_freq"] <= 1:
        raise ValueError("log_freq should be greater than 0")
    hp["log_freq"] = min(int(hp["log_freq"]), hp["num_rounds"])
    return hp