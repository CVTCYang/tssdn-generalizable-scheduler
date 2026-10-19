from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from .environment import TASEnvironment


@dataclass(frozen=True)
class ModelConfig:
    graph_dim: int = 64
    graph_layers: int = 2
    attention_heads: int = 4
    route_dim: int = 16
    lstm_dim: int = 64
    selector_dim: int = 256
    time_scale_ns: float = 1_000_000.0
    rate_scale_bps: float = 1_000_000_000.0
    packet_scale_bytes: float = 1500.0
    gumbel_temperature: float = 1.0

    def __post_init__(self):
        sizes = (self.graph_dim, self.graph_layers, self.attention_heads,
                 self.route_dim, self.lstm_dim, self.selector_dim)
        if any(v <= 0 for v in sizes) or self.graph_dim % self.attention_heads:
            raise ValueError("Positive dimensions and graph_dim divisible by heads are required")
        if min(self.time_scale_ns, self.rate_scale_bps, self.packet_scale_bytes,
               self.gumbel_temperature) <= 0:
            raise ValueError("Normalization scales and temperature must be positive")


class GraphAttention(MessagePassing):
    def __init__(self, input_dim, output_dim, heads):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.width = heads, output_dim // heads
        self.node = nn.Linear(input_dim, output_dim, bias=False)
        self.edge = nn.Linear(1, output_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 3 * self.width))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, nodes, source, target, rates):
        n = nodes.shape[0]
        h = self.node(nodes).view(n, self.heads, self.width)
        edge = self.edge(rates).view(-1, self.heads, self.width)
        edge_index = torch.stack((source, target), dim=0)
        aggregate = self.propagate(edge_index, x=h, edge_attr=edge, size=(n, n))
        return F.leaky_relu(aggregate.flatten(1) + self.bias, 0.2)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        joined = torch.cat((x_i, x_j, edge_attr), dim=-1)
        logits = F.leaky_relu((joined * self.attention).sum(-1), 0.2)
        weights = softmax(logits, index, ptr=ptr, num_nodes=size_i, dim=0)
        return (x_j + edge_attr) * weights.unsqueeze(-1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph = nn.ModuleList([
            GraphAttention(6 if i == 0 else config.graph_dim,
                           config.graph_dim, config.attention_heads)
            for i in range(config.graph_layers)])
        self.route = nn.GRU(config.graph_dim, config.route_dim, batch_first=True)
        self.flow_dim = config.route_dim + 4
        self.flow_mlp = mlp(self.flow_dim, self.flow_dim, self.flow_dim)
        self.sequence = nn.LSTMCell(self.flow_dim + 1, config.lstm_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def tensor(self, values):
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def encode_graph(self, topology):
        lookup = {node: i for i, node in enumerate(topology.nodes)}
        n = len(lookup)
        incoming, outgoing = [0] * n, [0] * n
        in_rate, out_rate = [0.0] * n, [0.0] * n
        sources, targets, rates = [], [], []
        for edge in topology.links:
            s, t = lookup[edge.source], lookup[edge.target]
            rate = edge.rate_bps / self.config.rate_scale_bps
            outgoing[s] += 1
            incoming[t] += 1
            out_rate[s] += rate
            in_rate[t] += rate
            sources.append(s)
            targets.append(t)
            rates.append([math.log1p(rate)])
        features = [[math.log1p(incoming[i]), math.log1p(outgoing[i]),
                     math.log1p(in_rate[i]), math.log1p(out_rate[i]),
                     float(incoming[i] == 1 and outgoing[i] == 1), 1.0]
                    for i in range(n)]
        sources.extend(range(n))
        targets.extend(range(n))
        rates.extend([[0.0]] * n)
        source = torch.tensor(sources, dtype=torch.long, device=self.device)
        target = torch.tensor(targets, dtype=torch.long, device=self.device)
        h, edge_rates = self.tensor(features), self.tensor(rates)
        for layer in self.graph:
            h = layer(h, source, target, edge_rates)
        batch = torch.zeros(n, dtype=torch.long, device=self.device)
        return h, global_mean_pool(h, batch)[0], lookup

    def encode_flow(self, flow, topology, graph):
        nodes, _, lookup = graph
        c = self.config
        steps = [nodes[lookup[node]] for node in flow.route]
        _, hidden = self.route(torch.stack(steps).unsqueeze(0))

        period = -1.0 if flow.period_ns is None else math.log1p(float(flow.period_ns) / c.time_scale_ns)
        attrs = self.tensor([period, math.log1p(float(flow.baseline_ns) / c.time_scale_ns),
                             flow.packet_bytes / c.packet_scale_bytes,
                             math.log1p(float(flow.deadline_ns) / c.time_scale_ns)])
        return self.flow_mlp(torch.cat((hidden[0, 0], attrs)))

    def empty_sequence(self):
        zero = self.tensor([0.0] * self.config.lstm_dim)
        return zero, zero.clone()

    def commit(self, embedding, offset, hidden):
        value = self.tensor([float(offset) / self.config.time_scale_ns])
        return self.sequence(torch.cat((embedding, value)), hidden)

    def forward(self, observation):
        graph = self.encode_graph(observation.topology)
        hidden = self.empty_sequence()
        for flow, offset in observation.committed:
            encoded = self.encode_flow(flow, observation.topology, graph)
            hidden = self.commit(encoded, offset, hidden)
        candidates = torch.stack([self.encode_flow(f, observation.topology, graph)
                                  for f in observation.remaining])
        return torch.cat((graph[1], hidden[0])), candidates


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        width = config.graph_dim + config.lstm_dim + self.encoder.flow_dim
        self.flow_selector = mlp(width, config.selector_dim, 1)
        self.offset_selector = mlp(width, config.selector_dim, 1)

    def decide(self, state, candidates, bounds, explore=False):
        repeated = state.expand(candidates.shape[0], -1)
        logits = self.flow_selector(torch.cat((repeated, candidates), -1)).squeeze(-1)
        if explore:
            selection = F.gumbel_softmax(logits, tau=self.config.gumbel_temperature,
                                        hard=True, dim=0)
        else:
            selection = F.one_hot(logits.argmax(), logits.numel()).to(logits.dtype)
        selected = selection @ candidates

        fraction = torch.sigmoid(self.offset_selector(torch.cat((state, selected))).squeeze(-1))
        offset = fraction * (selection @ bounds)
        return selection, offset

    def bounds(self, observation):
        env = TASEnvironment(observation.topology)
        return self.encoder.tensor([float(env.offset_bound_ns(f)) / self.config.time_scale_ns
                                    for f in observation.remaining])

    def forward(self, observation, explore=False):
        state, candidates = self.encoder(observation)
        return self.decide(state, candidates, self.bounds(observation), explore)


class Critic(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.encoder = Encoder(config)
        self.value = mlp(config.graph_dim + config.lstm_dim + self.encoder.flow_dim + 1,
                         config.selector_dim, 1)

    def forward(self, observation, selection, offset):
        state, candidates = self.encoder(observation)
        selected = selection @ candidates
        return self.value(torch.cat((state, selected, offset.reshape(1)))).squeeze(-1)


class Scheduler:

    def __init__(self, actor, environment):
        self.actor = actor
        self.env = environment
        self._encoded = {}
        self._prefix = ()
        with torch.no_grad():
            self.graph = actor.encoder.encode_graph(environment.topology)
            self.hidden = actor.encoder.empty_sequence()

    @torch.no_grad()
    def _flow(self, flow):
        if flow not in self._encoded:
            self._encoded[flow] = self.actor.encoder.encode_flow(flow, self.env.topology, self.graph)
        return self._encoded[flow]

    @torch.no_grad()
    def _sync(self):
        committed = self.env.observation.committed
        if committed[:len(self._prefix)] != self._prefix:
            raise ValueError("Committed offsets changed; create a new Scheduler session")
        for flow, offset in committed[len(self._prefix):]:
            self.hidden = self.actor.encoder.commit(self._flow(flow), offset, self.hidden)
        self._prefix = committed

    @torch.no_grad()
    def step(self, explore=False, offset_noise=0.0):
        self._sync()
        observation = self.env.observation
        if not observation.remaining:
            raise ValueError("No pending flows")
        state = torch.cat((self.graph[1], self.hidden[0]))
        candidates = torch.stack([self._flow(f) for f in observation.remaining])
        selection, offset = self.actor.decide(state, candidates,
                                              self.actor.bounds(observation), explore)
        index = int(selection.argmax())
        normalized = float(offset) + offset_noise
        result = self.env.step(observation.remaining[index].id,
                               normalized * self.actor.config.time_scale_ns)
        self._sync()
        return result, index, normalized

    def schedule(self, arrivals=()):
        self.env.add_flows(arrivals)
        results = []
        while self.env.observation.remaining:
            results.append(self.step()[0])
        return results


def save_actor(actor, path):
    torch.save({"config": asdict(actor.config), "actor": actor.state_dict()}, path)


def load_actor(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    actor = Actor(ModelConfig(**data["config"])).to(device)
    actor.load_state_dict(data["actor"])
    return actor.eval()
