# FLMMS: Federated Learning Multi-Machine Simulator
FLMMS is a Docker-based federated learning framework for simulating multi-machine training.

## Usage

### Quick Start

#### 1. Prepare Docker Environment

In directory `setup_utils`, run the following command to prepare the Docker environment:

```bash
./setup.bash
```

#### 2. Launch the Simulator

In directory `FLMMS`, run the following command to launch the simulator:

```bash
python launch.py
```

### Config

In [`FLMMS/configs/config.py`](FLMMS/configs/config.py), you can specify:

#### 1. Global Configurations

In the `GlobalConfig` class, you can specify:

- `expt_name`: The name of the experiment. Default is `None`, which will make the experiment results folder named for the current time like `240101114514`.
- `data_path`: The path to the dataset. Default is `data/`.
- `results_path`: The path to the experiment results. Default is `results/`.
- `log_level`: The log level. Default is `INFO`.
- `random_seed`: The random seed. Default is `42`.
- `dataloader_workers`: The number of workers for the dataloader. Default is `4`.
- `device`: The device to use. You can specify `cuda` or `cpu`. Default is `cpu`.
- `cuda_device`: available when `device` is `cuda`. Default is [0, 1, 2, 3], which means server uses GPU 0 and client 1 uses GPU 1, client 2 uses GPU 2, client 3 uses GPU 3.
- `num_clients`: The number of clients. Default is `3`. As the server is also a node, the total number of nodes is `num_clients + 1`.
- `data_distribution`: The distribution of the dataset. See code in [`FLMMS/datasets/datatool.py`](FLMMS/datasets/datatool.py) for details.

#### 2. Model Configurations

In the `ModelConfig` class, you can specify:

- `optimizer`: The optimizer.
- `scheduler`: The scheduler. Default is a dict `{ "name": "StepLR", "param": { "step_size": 1, "gamma": 0.5 } }` which specifies the `StepLR` scheduler with `step_size=1` and `gamma=0.5`.
- `lr`: The learning rate.
- `min_lr`: The minimum learning rate.
- `batchsize`: The batch size.

#### 3. experiment configurations

In the `ExptGroupConfig1` class, each parameter is a list of values. The simulator will run the experiment for each combination of the parameters.

e.g. when you specify:

```python
iteration = [100, 1000, 10000]
algo = [{"name": "None", "param": {}}, {"name": "FedAvg", "param": {"K": 5}}]
```

the simulator will run the experiment for 6 times, with the following configurations:

- iteration=100, algo=None
- iteration=100, algo=FedAvg
- iteration=1000, algo=None
- iteration=1000, algo=FedAvg
- iteration=10000, algo=None
- iteration=10000, algo=FedAvg

There are 2 default classed `ExptGroupConfig1` and `ExptGroupConfig2` in the `config.py` file. You can define your own experiment configurations.

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