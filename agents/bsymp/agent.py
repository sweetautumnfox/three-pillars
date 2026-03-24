from __future__ import annotations

import numpy as np

from .config import BSyMPConfig
from .model import BSyMPModel
from .neurogenesis import NeurogenesisStrategy, DefaultNeurogenesis


class BSyMPAgent:
    """BSyMP agent for the Conditional MAB environment.

    Encodes environment observations as a binary vector, runs BSyMPModel
    inference / learning, and decodes the action from hidden-state
    posteriors of the designated action nodes.
    """

    def __init__(
        self,
        config: BSyMPConfig,
        neurogenesis: NeurogenesisStrategy | None = None,
    ):
        self.config = config
        self.n_lights = config.n_lights
        self.n_arms = config.n_arms

        # Observation: lights (binary) + arm_signal (binary)
        self.N_o = config.n_lights + 1

        # Hidden states: n_arms action nodes + extra internal nodes
        N_s_init = config.n_arms + config.n_extra_states

        self.model = BSyMPModel(self.N_o, N_s_init, config)
        self.neurogenesis = neurogenesis or DefaultNeurogenesis()
        self.rng = np.random.default_rng(config.seed)

        self.total_steps = 0

    def encode_observation(self, obs: dict) -> np.ndarray:
        """Convert environment observation dict to binary o_t vector.

        Layout: ``[light_0, ..., light_{n-1}, arm_signal]``
        """
        o = np.zeros(self.N_o, dtype=np.float64)
        o[: self.n_lights] = obs["lights"]
        o[self.n_lights] = float(obs["arm_signal"])
        return o

    def select_action(self, s_t: np.ndarray) -> np.ndarray:
        """Pick arm from action-node posteriors (argmax)."""
        action_posteriors = s_t[: self.n_arms]
        chosen_arm = int(np.argmax(action_posteriors))
        action = np.zeros(self.n_arms, dtype=np.int8)
        action[chosen_arm] = 1
        return action

    def step(self, obs: dict) -> np.ndarray:
        """Full agent step: encode, infer, learn, act.

        Args:
            obs: environment observation dict with keys
                 ``lights``, ``arm_signal``, ``selected_arm``.

        Returns:
            MultiBinary action for the environment.
        """
        o_t = self.encode_observation(obs)
        s_t = self.model.step(o_t)
        action = self.select_action(s_t)

        self.total_steps += 1

        self.neurogenesis.maybe_add_states(
            self.model, self.config, self.rng, self.total_steps
        )

        return action
