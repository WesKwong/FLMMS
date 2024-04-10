from loguru import logger
import os
import pickle

import torch
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor

from configs.config import global_config as config


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
            f"Rank: {dist.get_rank()} | World Size: {dist.get_world_size()}")


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


def broadcast(data, dsts: list, tag=0, enable_async=config.async_comm) -> None:
    '''
    Broadcast data to all the destination processes.
    data: the data to be broadcasted.
    dsts: A list of destination processes.
    tag: the tag of the message.
    enable_async: whether to use async communication.
    '''
    if enable_async:
        with ThreadPoolExecutor() as executor:
            for dst in dsts:
                executor.submit(send, data, dst, tag)
    else:
        for dst in dsts:
            send(data, dst, tag)


def gather(srcs: list, tag=0, enable_async=config.async_comm) -> list:
    '''
    Gather data from all the source processes.
    srcs: A list of source processes.
    tag: the tag of the message.
    enable_async: whether to use async communication.
    '''
    data = []
    if enable_async:
        with ThreadPoolExecutor() as executor:
            for src in srcs:
                data.append(executor.submit(recv, src, tag))
        data = [future.result() for future in data]
    else:
        for src in srcs:
            data.append(recv(src, tag))
    return data


def scatter(data: list, dsts: list, tag=0, enable_async=config.async_comm):
    '''
    Scatter data to all the destination processes.
    data: A list of data to be scattered.
    dsts: A list of destination processes.
    tag: the tag of the message.
    enable_async: whether to use async communication.
    data[i] will be sent to dsts[i].
    '''
    if enable_async:
        with ThreadPoolExecutor() as executor:
            for i in range(len(dsts)):
                executor.submit(send, data[i], dsts[i], tag)
    else:
        for i in range(len(dsts)):
            send(data[i], dsts[i], tag)