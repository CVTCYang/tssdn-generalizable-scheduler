from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from types import MappingProxyType
from typing import Optional


def rational(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    rate_bps: int = 1_000_000_000

    def __post_init__(self):
        if self.rate_bps <= 0 or self.source == self.target:
            raise ValueError("Links require positive rates and distinct endpoints")


@dataclass(frozen=True)
class Topology:
    nodes: tuple[str, ...]
    links: tuple[Link, ...]
    switch_delay_ns: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "switch_delay_ns", rational(self.switch_delay_ns))
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Topology requires unique, nonempty nodes")
        if self.switch_delay_ns < 0:
            raise ValueError("Switch delay must be nonnegative")
        pairs = [(e.source, e.target) for e in self.links]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Parallel directed links are not supported")
        if any(a not in self.nodes or b not in self.nodes for a, b in pairs):
            raise ValueError("Unknown link endpoint")

    @property
    def rates(self):
        return {(e.source, e.target): e.rate_bps for e in self.links}


@dataclass(frozen=True)
class Flow:
    id: str
    route: tuple[str, ...]
    packet_bytes: int
    period_ns: Optional[Fraction]
    deadline_ns: Fraction
    baseline_ns: Fraction = Fraction(0)

    def __post_init__(self):
        object.__setattr__(self, "route", tuple(self.route))
        for key in ("deadline_ns", "baseline_ns", "period_ns"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, rational(getattr(self, key)))
        if not self.id or self.packet_bytes <= 0 or self.deadline_ns <= 0:
            raise ValueError("Invalid flow id, packet length or deadline")
        if self.period_ns is not None and self.period_ns <= 0:
            raise ValueError("Period must be positive; use None for a burst")
        if self.baseline_ns < 0 or len(self.route) < 2:
            raise ValueError("Invalid baseline or route")
        if len(set(self.route)) != len(self.route):
            raise ValueError("Fixed routes must be loop-free")


@dataclass(frozen=True)
class Window:
    link: tuple[str, str]
    start: Fraction
    duration: Fraction
    period: Optional[Fraction]
    flow_id: str

    @property
    def end(self):
        return self.start + self.duration


def windows_overlap(a: Window, b: Window) -> bool:
    if a.link != b.link:
        return False
    if a.period is None and b.period is None:
        return a.start < b.end and b.start < a.end
    if a.period is not None and b.period is not None:
        denominator = lcm(a.period.denominator, b.period.denominator)
        pa, pb = int(a.period * denominator), int(b.period * denominator)
        phase_cycle = Fraction(gcd(pa, pb), denominator)
        phase = (b.start - a.start) % phase_cycle
        return phase < a.duration or phase_cycle - phase < b.duration
    recurring, single = (a, b) if a.period is not None else (b, a)
    lower = (single.start - recurring.end) / recurring.period
    upper = (single.end - recurring.start) / recurring.period
    first_k = max(0, lower.numerator // lower.denominator + 1)
    return first_k < upper


@dataclass(frozen=True)
class Observation:
    topology: Topology
    remaining: tuple[Flow, ...]
    committed: tuple[tuple[Flow, Fraction], ...]


@dataclass(frozen=True)
class StepResult:
    flow_id: str
    offset_ns: Fraction
    accepted: bool
    reward: float
    reason: str
    observation: Observation
    done: bool


class TASEnvironment:

    def __init__(self, topology: Topology, flows=()):
        self.topology = topology
        self._rates = topology.rates
        self._remaining = {}
        self._committed = {}
        self._rejected = {}
        self._seen = set()
        self._windows = {key: [] for key in self._rates}
        self.processing_order = []
        self.add_flows(flows)

    @property
    def committed(self):
        return MappingProxyType(self._committed)

    @property
    def rejected(self):
        return MappingProxyType(self._rejected)

    @property
    def observation(self):
        return Observation(self.topology, tuple(self._remaining.values()),
                           tuple(self._committed.values()))

    @property
    def schedulability(self):
        processed = len(self._committed) + len(self._rejected)
        return len(self._committed) / processed if processed else 0.0

    def add_flows(self, flows):
        flows = tuple(flows)
        ids = [flow.id for flow in flows]
        if len(set(ids)) != len(ids) or self._seen.intersection(ids):
            raise ValueError("Flow ids must be unique across all arrivals")
        for flow in flows:
            if any(link not in self._rates for link in zip(flow.route, flow.route[1:])):
                raise ValueError(f"Flow {flow.id}: route uses an absent directed link")
        self._remaining.update((flow.id, flow) for flow in flows)
        self._seen.update(ids)

    def transmission_ns(self, flow):
        return sum((Fraction(flow.packet_bytes * 8 * 10**9, self._rates[link])
                    for link in zip(flow.route, flow.route[1:])), Fraction(0))

    def path_latency_ns(self, flow):
        return self.transmission_ns(flow) + max(0, len(flow.route) - 2) * self.topology.switch_delay_ns

    def offset_bound_ns(self, flow):
        bound = max(Fraction(0), flow.deadline_ns - self.path_latency_ns(flow))
        return min(bound, flow.period_ns) if flow.period_ns is not None else bound

    def windows(self, flow, offset_ns):
        start = flow.baseline_ns + rational(offset_ns)
        result = []
        for link in zip(flow.route, flow.route[1:]):
            duration = Fraction(flow.packet_bytes * 8 * 10**9, self._rates[link])
            result.append(Window(link, start, duration, flow.period_ns, flow.id))
            start += duration + self.topology.switch_delay_ns
        return tuple(result)

    def validate(self, flow, offset_ns):
        offset = rational(offset_ns)
        if offset < 0 or (flow.period_ns is not None and offset >= flow.period_ns):
            return False, "offset-domain"
        if offset + self.path_latency_ns(flow) > flow.deadline_ns:
            return False, "deadline"
        for window in self.windows(flow, offset):
            if window.period is not None and window.duration > window.period:
                return False, "self-overlap"
            for prior in self._windows[window.link]:
                if windows_overlap(window, prior):
                    return False, f"collision:{window.link[0]}->{window.link[1]}:{prior.flow_id}"
        return True, "accepted"

    def step(self, flow_id, offset_ns):
        flow = self._remaining[flow_id]
        offset = rational(offset_ns)
        valid, reason = self.validate(flow, offset)
        del self._remaining[flow_id]
        self.processing_order.append(flow_id)
        if valid:
            self._committed[flow_id] = (flow, offset)
            for window in self.windows(flow, offset):
                self._windows[window.link].append(window)
            reward = 1 - float((offset + self.path_latency_ns(flow)) / flow.deadline_ns)
        else:
            self._rejected[flow_id] = reason
            reward = -5.0
        return StepResult(flow_id, offset, valid, reward, reason, self.observation,
                          not self._remaining)

    def gate_windows(self, start_ns, end_ns, max_windows=1_000_000):
        start, end = rational(start_ns), rational(end_ns)
        if end <= start:
            raise ValueError("Export interval must have positive duration")
        result = []
        for windows in self._windows.values():
            for w in windows:
                k = 0 if w.period is None else max(0, (start - w.end) // w.period + 1)
                while True:
                    left = w.start + (k * w.period if w.period is not None else 0)
                    if left >= end:
                        break
                    if left + w.duration > start:
                        if len(result) >= max_windows:
                            raise ValueError("Gate export exceeds max_windows")
                        result.append(Window(w.link, left, w.duration, None, w.flow_id))
                    if w.period is None:
                        break
                    k += 1
        return sorted(result, key=lambda w: (w.start, w.link, w.flow_id))
