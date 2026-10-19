# O4AS

An incremental no-wait TAS scheduler with graph and recurrent neural encoders.

## Installation

Requires Python 3.10+, PyTorch 2.2+, and PyTorch Geometric 2.5+.

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Quick Start

From the repository directory, install the dependencies above and run:

```sh
python quick_start.py
```

The example builds a three-node topology in memory, performs four small training episodes, schedules two periodic flows, and admits one additional one-shot flow without changing existing offsets. It prints training rewards and admission decisions. No dataset or checkpoint download is required, and no files are generated. This is a functional demonstration, not a trained performance benchmark.

For an existing checkpoint, use `load_actor("checkpoint.pt")` with `Scheduler` as shown below. To train on your own traffic, supply `(topology, flows)` pairs to `Trainer.episode()`.

## Modules

- `o4as.environment`: fixed-route scheduling, exact periodic link-window collision checks, deadlines, and incremental admission.
- `o4as.model`: edge-aware graph attention with LeakyReLU, PyTorch Geometric `MessagePassing`, neighborhood `softmax`, and `global_mean_pool`; a GRU over the complete route-node sequence; an MLP over concatenated route and scalar features; committed-flow LSTM, flow selection, and offset prediction.
- `o4as.training`: DDPG training, replay, target networks, and checkpoint management.

Times are expressed in nanoseconds, packet lengths in bytes, and link rates in bits per second. Routes use directed links. Set `period_ns=None` for a one-shot flow. Previously committed offsets remain unchanged when new flows arrive.

## API

Construct a `Topology` from nodes and `Link` objects, and represent traffic using `Flow` objects. Given `topology`, `initial_flows`, and `new_flows`:

```python
from o4as import TASEnvironment
from o4as.model import Scheduler, load_actor

env = TASEnvironment(topology, initial_flows)
scheduler = Scheduler(load_actor("checkpoint.pt"), env)
results = scheduler.schedule()
new_results = scheduler.schedule(new_flows)
```

Train using externally supplied scenarios:

```python
from o4as.training import Trainer

trainer = Trainer()
for topology, flows in training_scenarios:
    metrics = trainer.episode(topology, flows)
trainer.save("checkpoint.pt")
```

`ModelConfig` and `TrainingConfig` expose network and training parameters. Model checkpoints are compatible with `load_actor`. The repository contains source code only; datasets, experiment results, and trained weights are not included.
