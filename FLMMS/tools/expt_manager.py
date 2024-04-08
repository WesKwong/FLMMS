from loguru import logger
# ------------------------------------------------------ #
import os
import numpy as np

def save_results(results_dict, path, name, verbose=True):
    save_path = os.path.join(path, name)
    results_numpy = np.array(results_dict)

    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(save_path, results=results_numpy)

    if verbose:
        logger.info(f"Saved results to {save_path+'.npz'}")


def load_results(path, filename, verbose=True):
    results_numpy = np.load(path+filename, allow_pickle=True)

    results_dict = results_numpy['results'].item()

    if verbose:
        logger.info(f"Loaded results from {path+filename}")

    return results_dict


def create_experiments(expt_groups_configs, verbose=True):
    log_id = 0
    expt_groups = dict()
    for name, expt_group_configs in expt_groups_configs.items():
        expts = list()
        for hp in expt_group_configs:
            expts.append(Experiment(hyperparameters=hp, log_id=log_id))
            log_id += 1
        expt_groups[name] = expts
    if verbose:
        logger.info(f"{len(expt_groups)} experiment groups created")
        total_expts = 0
        for name, expts in expt_groups.items():
            logger.info(f"Group {name}: {len(expts)} experiments")
            total_expts += len(expts)
        logger.info(f"Total experiments: {total_expts}")
    return expt_groups


class Experiment():
    hyperparameters = {}
    results = {}

    def __init__(self, hyperparameters, log_id=None):
        self.hyperparameters = hyperparameters
        self.results = {}
        self.hyperparameters['log_id'] = log_id
        if log_id is None:
            self.hyperparameters['log_id'] = np.random.randint(100000)

    def update_hp(self, hp):
        self.hyperparameters.update(hp)

    def log_hp(self):
        logger.info("Hyperparameters:")
        for key, value in self.hyperparameters.items():
            logger.info(" - "+key+" "*(24-len(key))+str(value))

    def log(self, update_dict, printout=True, override=False):
        # update result
        for key, value in update_dict.items():
            if (not key in self.results) or override:
                self.results[key] = [value]
            else:
                self.results[key] += [value]

        if printout:
            logger.info(update_dict)

    def is_log_round(self, c_round):
        log_freq = self.hyperparameters['log_freq']
        return (c_round == 1) or (c_round % log_freq == 0) or (c_round == self.hyperparameters['iteration'])

    def to_dict(self):
        # turns an experiment into a dict that can be saved to disc
        expt_dict = {"hp": self.hyperparameters,
                     **self.results}
        return expt_dict

    def save_to_disc(self, path):
        save_results(self.to_dict(), path, 'expt_'+str(self.hyperparameters['log_id']))