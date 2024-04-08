# FLMMS: Federated Learning Multi-Machine Simulator
FLMMS is a Docker-based federated learning framework for simulating multi-machine training.

## Requirements

Conda is recommended for managing the Python environment. The authors' environment is set up as follows:

### 1. Create a conda environment and activate it

```bash
conda create -n flmms python=3.9 -y
```

### 2. Install the required packages

#### about torch

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
```

#### about other packages

```bash
conda install numpy tqdm loguru -y
```

## Usage

### Quick Start

#### 1. Prepare docker environment

In directory `setup_utils`, run the following command to prepare the Docker environment:

```bash
./setup.bash
```

#### 2. Launch the simulator

In directory `FLMMS`, run the following command to launch the simulator:

```bash
python launch.py
```

### Configurations

In [`FLMMS/configs/config.py`](FLMMS/configs/config.py), you can specify:

#### 1. Global configurations

In the `GlobalConfig` class, you can specify:

- `expt_name`: The name of the experiment. Default is `None`, which will make the experiment results folder named for the current time like `240101114514`.
- `data_path`: The path to the dataset. Default is `data/`.
- `results_path`: The path to save the experiment results and node running logs. Default is `results/`.
- `random_seed`: The random seed. Default is `42`.
- `monitor_server_log`: Whether or not to monitor the server's log in the terminal after launching the simulator. The default is `True`. If set it to `False`, you can monitor the server's log in the `results` folder. In addition to this, you can also monitor the logs of all clients in the `results` folder.
- `log_level`: The log level. Default is `INFO`.
- `dataloader_workers`: The number of workers for the dataloader. Default is `4`.
- `device`: The device to use. You can specify `cuda` or `cpu`. Default is `cpu`.
- `cuda_device`: available when `device` is `cuda`. Default is [0, 1, 2, 3], which means server uses GPU 0 and client 1 uses GPU 1, client 2 uses GPU 2, client 3 uses GPU 3.
- `num_client`: The number of clients. Default is `3`. As the server is also a node, the total number of nodes is `num_client + 1`.
- `data_distribution`: The distribution of the dataset. See code in [`FLMMS/datasets/datatool.py`](FLMMS/datasets/datatool.py) for details.
- `enable_prepare_dataset`: Whether or not to prepare the dataset. Default is `True`. A recommended way is set it to `True` when you change the `num_client` or `data_distribution`, and set it to `False` in next experiments used the same `num_client` and `data_distribution`.

#### 2. Model configurations

In the `ModelConfig` class, you can specify:

- `optimizer`: The optimizer.
- `scheduler`: The scheduler. Default is a dict `{ "name": "StepLR", "param": { "step_size": 1, "gamma": 0.5 } }` which specifies the `StepLR` scheduler with `step_size=1` and `gamma=0.5`.
- `lr`: The learning rate.
- `min_lr`: The minimum learning rate.
- `batchsize`: The batch size.

#### 3. Experiment configurations

In the `ExptGroupConfig1` class, each parameter is a list of values. The simulator will run the experiment for each combination of the parameters.

e.g. When you specify:

```python
iteration = [100, 1000, 10000]
algo = [{"name": "None", "param": {}}, {"name": "FedAvg", "param": {"K": 5}}]
```

The simulator will run the experiment for 6 times, with the following configurations:

- `iteration=100, algo=None`
- `iteration=100, algo=FedAvg`
- `iteration=1000, algo=None`
- `iteration=1000, algo=FedAvg`
- `iteration=10000, algo=None`
- `iteration=10000, algo=FedAvg`

There are 2 default classes `ExptGroupConfig1` and `ExptGroupConfig2` in the `config.py` file. You can define your own experiment configurations.

You can specify the following parameters:

- `group_name`: The name of the experiment group.
- `dataset`: The dataset to use.
- `net`: The neural network model to use.
- `iteration`: The number of iterations.
- `algo`: The federated learning algorithm to use.
- `log_freq`: The frequency of logging.

## Acknowledgments

This project was inspired by the following projects:

- [felisat/federated-learning](https://github.com/felisat/federated-learning)
- [Jiang-Yutong/DAGC](https://github.com/Jiang-Yutong/DAGC)