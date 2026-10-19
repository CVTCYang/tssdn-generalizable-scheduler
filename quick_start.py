import torch

from o4as import Flow, Link, Topology, TASEnvironment
from o4as.model import Scheduler
from o4as.training import Trainer, TrainingConfig


def main():
    torch.set_num_threads(1)
    topology = Topology(
        ("A", "B", "C"),
        (Link("A", "B"), Link("B", "C")),
    )
    flows = (
        Flow("f0", ("A", "B", "C"), 100, 1_000_000, 1_000_000),
        Flow("f1", ("A", "B", "C"), 200, 1_000_000, 1_000_000),
    )
    trainer = Trainer(config=TrainingConfig(batch_size=2, replay_capacity=32))
    for _ in range(4):
        metrics = trainer.episode(topology, flows)
        print(f"Episode {metrics['episode']}: reward={metrics['reward']:.3f}")

    env = TASEnvironment(topology, flows)
    scheduler = Scheduler(trainer.actor.eval(), env)
    results = scheduler.schedule()
    previous = env.observation.committed
    arriving = Flow("f2", ("A", "B", "C"), 100, None, 1_000_000, 2_000_000)
    results.extend(scheduler.schedule([arriving]))
    assert env.observation.committed[:len(previous)] == previous
    for result in results:
        print(f"{result.flow_id}: accepted={result.accepted}, "
              f"offset_ns={float(result.offset_ns):.3f}, reason={result.reason}")
    print(f"Accepted: {len(env.committed)}, rejected: {len(env.rejected)}")


if __name__ == "__main__":
    main()
