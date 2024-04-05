# FLMMS: Federated Learning Multi-Machine Simulator
FLMMS is a Docker-based federated learning framework for simulating multi-machine training.

## Usage

### Quick Start

#### 1. Get Docker Image

```bash
cd SetupUtils && ./GetImage.bash
```

#### 2. Create a Docker Network

```bash
cd SetupUtils && ./CreateDockerNet.bash
```

#### 3. Launch the Simulator

```bash
cd src && python launch.py
```
