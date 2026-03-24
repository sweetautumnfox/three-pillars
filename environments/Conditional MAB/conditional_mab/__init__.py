from .config import ConditionalMABConfig
from .env import ConditionalMABEnv
from .rule import evaluate_rule, generate_random_rule

import gymnasium

gymnasium.register(
    id="ConditionalMAB-v0",
    entry_point="conditional_mab.env:ConditionalMABEnv",
    max_episode_steps=None,
)

__all__ = [
    "ConditionalMABConfig",
    "ConditionalMABEnv",
    "evaluate_rule",
    "generate_random_rule",
]
