import itertools

import numpy as np


def generate_random_rule(
    n_lights: int, n_arms: int, rng: np.random.Generator
) -> dict[tuple[int, ...], int]:
    """Generate a random lookup table mapping each light config to a correct arm.

    For each of the 2^n_lights possible binary configurations, uniformly
    samples one of the n_arms arms as the correct choice.
    """
    rule = {}
    for config in itertools.product([0, 1], repeat=n_lights):
        rule[config] = int(rng.integers(0, n_arms))
    return rule


def evaluate_rule(rule: dict[tuple[int, ...], int], light_config: tuple[int, ...]) -> int:
    """Look up the correct arm for a given light configuration."""
    return rule[light_config]
