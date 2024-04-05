import os
from loguru import logger

from configs.MainConfig import config
from tools.ExptUtils import get_unique_results_path

NETWORK_NAME = "flmms_net"
IMAGE_NAME = "weskwong/flmms:v1.0"
MNT_PATH = "$(pwd)"
RESULTS_PATH = get_unique_results_path(config.results_path, config.expt_name)

MASTER_ADDR = "flmms_server"
MASTER_PORT = "11451"
WORLD_SIZE = config.num_client + 1
DEVICE = config.device


def get_torchrun_cmd(id):
    nproc_per_node = f"--nproc_per_node=1"
    nnodes = f"--nnodes={WORLD_SIZE}"
    node_rank = f"--node_rank={id}"
    master_addr = f"--master_addr={MASTER_ADDR}"
    master_port = f"--master_port={MASTER_PORT}"
    torchrun_cmd = f"torchrun {nproc_per_node} {nnodes} {node_rank} {master_addr} {master_port} run.py"
    return torchrun_cmd


def run_docker(id):
    # docker command
    docker_run_mode = "-d"
    network = f"--network={NETWORK_NAME}"
    mnt = f"-v {MNT_PATH}:/workspace"
    image = f"{IMAGE_NAME}"
    if id == 0:
        container_name = f"--name=flmms_server"
    else:
        container_name = f"--name=flmms_client_{id}"
    if DEVICE == "cuda":
        gpu_cmd = "--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    else:
        gpu_cmd = ""
    # run script
    torchrun_cmd = get_torchrun_cmd(id)
    if id == 0:
        log_name = os.path.join(RESULTS_PATH, "server.log")
    else:
        log_name = os.path.join(RESULTS_PATH, f"client_{id}.log")
    sh_cmd = f"CUDA_VISIBLE_DEVICES={config.cuda_device[id]} RESULTS_PATH={RESULTS_PATH} {torchrun_cmd} > {log_name} 2>&1"
    sh_cmd = f"sh -c '{sh_cmd}'"
    # run docker
    docker_cmd = f"docker run --rm {docker_run_mode} {network} {mnt} {container_name} {gpu_cmd} {image} {sh_cmd}"
    os.system(docker_cmd)


@logger.catch
def launch():
    run_docker(0)
    for i in range(1, config.num_client+1):
        run_docker(i)


if __name__ == "__main__":
    launch()