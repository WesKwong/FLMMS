from .server import *
from .client import *

def get_server_runner(hp):
    algo = hp["algo"]["name"]
    if algo == "None":
        run_func = base_run_server
    elif algo == "FedAvg":
        run_func = base_run_server
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    return run_func

def get_client_runner(hp):
    algo = hp["algo"]["name"]
    if algo == "None":
        run_func = base_run_client
    elif algo == "FedAvg":
        run_func = base_run_client
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    return run_func