from .server import *
from .client import *

def get_server_runner(hp):
    algo = hp["algo"]["name"]
    if algo == "None":
        run_func = BaseServerRunner
    elif algo == "FedAvg":
        run_func = BaseServerRunner
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    return run_func

def get_client_runner(hp):
    algo = hp["algo"]["name"]
    if algo == "None":
        run_func = BaseClientRunner
    elif algo == "FedAvg":
        run_func = BaseClientRunner
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    return run_func