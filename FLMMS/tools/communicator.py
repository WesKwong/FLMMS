from loguru import logger
import os
import pickle

import torch
import torch.distributed as dist


def init_communication_group(verbose=True) -> None:
    '''
    Initialize the communication group for the distributed training.
    '''
    logger.info("Initializing Communication Group...")
    dist.init_process_group(backend="gloo")
    logger.info("Initialized Done!")
    if verbose:
        logger.info(
            f"Master Address: {os.environ['MASTER_ADDR']} | Master Port: {os.environ['MASTER_PORT']}"
        )
        logger.info(
            f"Rank/WorldSize: {dist.get_rank()}/{dist.get_world_size()}")


def destroy_communication_group() -> None:
    '''
    Destroy the communication group.
    '''
    logger.info("Destroying Communication Group...")
    dist.destroy_process_group()
    logger.info("Destroyed Done!")


def is_server() -> bool:
    '''
    Check if the current process is the server.
    '''
    return dist.get_rank() == 0


def send(data, dst: int, tag=0) -> None:
    '''
    Send data to the destination process.
    data: the data to be sent.
    dst: the destination process.
    tag: the tag of the message.
    '''
    serialized_data = pickle.dumps(data)
    data_tensor = torch.tensor(list(serialized_data),
                               dtype=torch.uint8).to(torch.device('cpu'))
    data_size = torch.tensor(len(data_tensor),
                             dtype=int).to(torch.device('cpu'))
    dist.send(data_size, dst=dst, tag=tag)
    dist.send(data_tensor, dst=dst, tag=tag)


def recv(src: int, tag=0) -> object:
    '''
    Receive data from the source process.
    src: the source process.
    tag: the tag of the message.
    '''
    data_size = torch.tensor(0, dtype=int).to(torch.device('cpu'))
    dist.recv(data_size, src=src, tag=tag)
    data_tensor = torch.empty(size=(data_size.item(), ),
                              dtype=torch.uint8).to(torch.device('cpu'))
    dist.recv(data_tensor, src=src, tag=tag)
    serialized_data = bytes(data_tensor.tolist())
    data = pickle.loads(serialized_data)
    return data


def broadcast(data, dsts: list, tag=0) -> None:
    '''
    Broadcast data to all the destination processes.
    data: the data to be broadcasted.
    dsts: A list of destination processes.
    tag: the tag of the message.
    '''
    for dst in dsts:
        send(data, dst, tag)


def gather(srcs: list, tag=0) -> list:
    '''
    Gather data from all the source processes.
    srcs: A list of source processes.
    tag: the tag of the message.
    '''
    data = []
    for src in srcs:
        data.append(recv(src, tag))
    return data


def scatter(data: list, dsts: list, tag=0):
    '''
    Scatter data to all the destination processes.
    data: A list of data to be scattered.
    dsts: A list of destination processes.
    tag: the tag of the message.
    data[i] will be sent to dsts[i].
    '''
    for i in range(len(data)):
        send(data[i], dsts[i], tag)
