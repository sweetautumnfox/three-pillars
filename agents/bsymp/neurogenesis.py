from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import BSyMPConfig

if __import__("typing").TYPE_CHECKING:
    from .model import BSyMPModel


class NeurogenesisStrategy(ABC):
    """Abstract base class for neurogenesis strategies.

    Subclass this and override ``maybe_add_states`` to implement a
    custom neurogenesis policy.
    """

    @abstractmethod
    def maybe_add_states(
        self,
        model: BSyMPModel,
        config: BSyMPConfig,
        rng: np.random.Generator,
        timestep: int,
    ) -> int:
        """Possibly add new hidden states to the model.

        Returns:
            Number of states added (0 if none).
        """
        ...


class DefaultNeurogenesis(NeurogenesisStrategy):
    """Stochastic neurogenesis: Bernoulli check each timestep.

    When triggered, adds ``config.neurogenesis_n_new`` hidden states
    with random initial C_A / C_B connectivity drawn uniform in [0.3, 0.7].
    """

    def maybe_add_states(
        self,
        model: BSyMPModel,
        config: BSyMPConfig,
        rng: np.random.Generator,
        timestep: int,
    ) -> int:
        if not config.neurogenesis_enabled:
            return 0
        if model.N_s >= config.neurogenesis_max_states:
            return 0
        if rng.random() > config.neurogenesis_p:
            return 0

        n_new = min(
            config.neurogenesis_n_new,
            config.neurogenesis_max_states - model.N_s,
        )

        # Random initial connectivity for new states
        c_A_new = rng.uniform(0.3, 0.7, size=(model.N_o, n_new))
        c_B_new_col = rng.uniform(0.3, 0.7, size=(model.N_s, n_new))
        c_B_new_row = rng.uniform(0.3, 0.7, size=(n_new, model.N_s + n_new))

        model.add_states(n_new, config, c_A_new=c_A_new,
                         c_B_new_col=c_B_new_col, c_B_new_row=c_B_new_row)
        return n_new
