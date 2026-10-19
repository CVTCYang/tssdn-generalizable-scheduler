from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import random
import torch
from torch.nn import functional as F
from .environment import TASEnvironment
from .model import Actor, Critic, ModelConfig, Scheduler


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    discount: float = 0.95
    target_tau: float = 0.005
    batch_size: int = 64
    replay_capacity: int = 100_000
    updates_per_episode: int = 1
    noise_scale: float = 0.4
    noise_decay: float = 0.98
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    grad_clip: float = 10.0
    seed: int = 42

    def __post_init__(self):
        if not 0 <= self.discount <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Invalid discount or target update coefficient")
        if min(self.batch_size, self.replay_capacity, self.updates_per_episode) < 1:
            raise ValueError("Batch, replay capacity and update count must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must cover a minibatch")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("Learning rate and gradient clip must be positive")
        if not 0 < self.noise_decay <= 1 or min(self.noise_scale, self.ou_theta, self.ou_sigma) < 0:
            raise ValueError("Invalid exploration configuration")


@dataclass(frozen=True)
class Transition:
    observation: object
    index: int
    offset: float
    reward: float
    next_observation: object
    done: bool


class Trainer:
    def __init__(self, model_config=ModelConfig(), config=TrainingConfig(), device="cpu"):
        self.config = config
        self.random = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.actor = Actor(model_config).to(device)
        self.critic = Critic(model_config).to(device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.episodes = 0

    def episode(self, topology, flows):
        env = TASEnvironment(topology, flows)
        scheduler = Scheduler(self.actor, env)
        noise, total_reward = 0.0, 0.0
        scale = self.config.noise_scale * self.config.noise_decay ** self.episodes
        while env.observation.remaining:
            observation = env.observation
            noise += -self.config.ou_theta * noise + self.config.ou_sigma * self.random.gauss(0, 1)
            result, index, offset = scheduler.step(explore=True, offset_noise=scale * noise)
            self.replay.append(Transition(observation, index, offset, result.reward,
                                          result.observation, result.done))
            total_reward += result.reward
        self.episodes += 1
        losses = {}
        if len(self.replay) >= self.config.batch_size:
            for _ in range(self.config.updates_per_episode):
                losses = self.update()
        return {"episode": self.episodes, "reward": total_reward,
                "schedulability": env.schedulability, "accepted": len(env.committed),
                "rejected": len(env.rejected), **losses}

    def update(self):
        batch = self.random.sample(list(self.replay), self.config.batch_size)
        predictions, targets = [], []
        device = self.actor.encoder.device
        for item in batch:
            selection = F.one_hot(torch.tensor(item.index, device=device),
                                  len(item.observation.remaining)).float()
            action = torch.tensor(item.offset, device=device)
            predictions.append(self.critic(item.observation, selection, action))
            with torch.no_grad():
                target = torch.tensor(item.reward, device=device)
                if not item.done:
                    next_selection, next_offset = self.target_actor(item.next_observation)
                    target = target + self.config.discount * self.target_critic(
                        item.next_observation, next_selection, next_offset)
                targets.append(target)
        critic_loss = F.mse_loss(torch.stack(predictions), torch.stack(targets))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        try:
            values = []
            for item in batch:
                selection, offset = self.actor(item.observation, explore=True)
                values.append(self.critic(item.observation, selection, offset))
            actor_loss = -torch.stack(values).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for target_net, online in ((self.target_actor, self.actor),
                                       (self.target_critic, self.critic)):
                for target, parameter in zip(target_net.parameters(), online.parameters()):
                    target.lerp_(parameter, self.config.target_tau)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach())}

    def save(self, path):
        torch.save({"config": asdict(self.actor.config), "training": asdict(self.config),
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "target_actor": self.target_actor.state_dict(),
                    "target_critic": self.target_critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "episodes": self.episodes, "random_state": self.random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(ModelConfig(**data["config"]), TrainingConfig(**data["training"]), device)
        for key in ("actor", "critic", "target_actor", "target_critic",
                    "actor_optimizer", "critic_optimizer"):
            getattr(trainer, key).load_state_dict(data[key])
        trainer.episodes = data["episodes"]
        trainer.random.setstate(data["random_state"])
        torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in data["cuda_rng"]])
        return trainer
