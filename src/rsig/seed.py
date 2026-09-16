from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def derive_seed(root_seed: int, *coordinates: int) -> int:
    values = (root_seed, *coordinates)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in values):
        raise ValueError("seed components must be non-negative integers")
    sequence = np.random.SeedSequence(int(root_seed), spawn_key=tuple(int(x) for x in coordinates))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True)
class ComponentSeeds:
    source: int
    modulation: int
    channel: int
    impairment: int
    noise: int


def component_seeds(sample_seed: int) -> ComponentSeeds:
    return ComponentSeeds(*(derive_seed(sample_seed, index) for index in range(5)))
