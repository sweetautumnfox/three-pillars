from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BSyMPConfig:
    """Configuration for the BSyMP agent."""

    # Environment dimensions (set from env)
    n_lights: int = 3
    n_arms: int = 4

    # Model dimensions
    n_extra_states: int = 0  # additional hidden states beyond n_arms

    # Prior concentrations
    a_prior: float = 1.0  # Dirichlet concentration for A (flat prior)
    b_prior: float = 1.0  # Dirichlet concentration for B (flat prior)

    # Initial connectivity (high = assume connections exist)
    c_A_init: float = 0.9
    c_B_init: float = 0.9

    # Session length for connectivity updates (Eqs 16-17)
    session_length: int = 10

    # Neurogenesis
    neurogenesis_enabled: bool = True
    neurogenesis_p: float = 0.02  # per-timestep probability
    neurogenesis_n_new: int = 1  # states added per event
    neurogenesis_max_states: int = 20  # upper bound on N_s

    # Run parameters
    max_trials: int | None = 500
    seed: int | None = None
    log_interval: int = 10
