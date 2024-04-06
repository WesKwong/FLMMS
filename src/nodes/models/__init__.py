from .client import *
from .server import *

def get_client_model(hp):
    algo = hp["algo"]
    if algo == "None":
        return BaseClientModel
    elif algo == "FedAvg":
        return BaseClientModel
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

def get_server_model(hp):
    algo = hp["algo"]
    if algo == "None":
        return BaseServerModel
    elif algo == "FedAvg":
        return BaseServerModel
    else:
        raise ValueError(f"Invalid algorithm: {algo}")