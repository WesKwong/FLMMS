import tools.GlobVarManager as glob
logger = glob.get('logger')
# ------------------------------------------------------ #
import os
import numpy as np

def save_results(results_dict, path, name, verbose=True):
    results_numpy = np.array(results_dict)

    if not os.path.exists(path):
        os.makedirs(path)

    np.savez(path+name, results=results_numpy)

    if verbose:
        logger.info(f"Saved results to {path+name+'.npz'}")


def load_results(path, filename, verbose=True):
    results_numpy = np.load(path+filename, allow_pickle=True)

    results_dict = results_numpy['results'].item()

    if verbose:
        logger.info(f"Loaded results from {path+filename}")

    return results_dict


def create_experiments(expt_groups_configs):
    log_id = 0
    expt_groups = dict()
    for name, expt_group_configs in expt_groups_configs.items():
        expts = list()
        for hp in expt_group_configs:
            expts.append(Experiment(hyperparameters=hp, log_id=log_id))
            log_id += 1
        expt_groups[name] = expts
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

    def __str__(self):
        selfname = "Hyperparameters: \n"
        for key, value in self.hyperparameters.items():
            selfname += " - "+key+" "*(24-len(key))+str(value)+"\n"
        return selfname

    def __repr__(self):
        return self.__str__()

    def log(self, update_dict, printout=True, override=False):
        # update result
        for key, value in update_dict.items():
            if (not key in self.results) or override:
                self.results[key] = [value]
            else:
                self.results[key] += [value]

        if printout:
            logger.info(update_dict)

    def to_dict(self):
        # turns an experiment into a dict that can be saved to disc
        expt_dict = {"hp": self.hyperparameters,
                     **self.results}
        return expt_dict

    def save_to_disc(self, path):
        save_results(self.to_dict(), path, 'expt_'+str(self.hyperparameters['log_id']))