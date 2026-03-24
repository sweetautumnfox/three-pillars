from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import ConditionalMABConfig
from .rule import generate_random_rule, evaluate_rule


class ConditionalMABEnv(gym.Env):
    """Non-stationary conditional deterministic Multi-Armed Bandit.

    Binary "lights" provide context each trial. The agent submits a MultiBinary
    action: all-zeros means "wait" (no-action), otherwise the first selected
    index is treated as the chosen arm. After ``max_steps_per_trial`` consecutive
    waits the environment forces a random arm selection.

    The agent receives a binary ``arm_signal`` (1 if correct, 0 otherwise) but
    **no reward** — learning must happen through the signal alone.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: ConditionalMABConfig | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = ConditionalMABConfig(**kwargs)
        config.validate()
        self.config = config

        self.render_mode = render_mode
        self.action_space = spaces.MultiBinary(config.n_arms)

        if config.flatten_obs:
            # Flat vector: [selected_arm, arm_signal, light1, ..., lightN]
            # selected_arm encoded as int in [-1, K-1], shifted to [0, K] for MultiDiscrete
            self.observation_space = spaces.MultiDiscrete(
                [config.n_arms + 1, 2] + [2] * config.n_lights
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "selected_arm": spaces.Discrete(config.n_arms + 1, start=-1),
                    "arm_signal": spaces.Discrete(2),
                    "lights": spaces.MultiBinary(config.n_lights),
                }
            )

        self.reward_range = (0.0, 0.0)

        # State (initialized in reset)
        self.current_rule: dict[tuple[int, ...], int] = {}
        self.current_lights: tuple[int, ...] = ()
        self.phase_id: int = 0
        self.trial_in_phase: int = 0
        self.total_trials: int = 0
        self.phase_duration: int = 0
        self.last_arm: int = -1
        self.last_signal: int = 0
        self.steps_in_trial: int = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_rule = generate_random_rule(
            self.config.n_lights, self.config.n_arms, self.np_random
        )
        self.phase_duration = self._sample_phase_duration()
        self.phase_id = 0
        self.trial_in_phase = 0
        self.total_trials = 0
        self.last_arm = -1
        self.last_signal = 0
        self.steps_in_trial = 0
        self.current_lights = self._sample_light_config()

        obs = self._build_observation(self.last_arm, self.last_signal, self.current_lights)
        info = self._build_info(phase_changed=False, forced=False)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int8)
        assert self.action_space.contains(action), f"Invalid action {action}"

        selected_indices = np.flatnonzero(action)

        # No-action (wait): all zeros
        if len(selected_indices) == 0:
            if self.steps_in_trial < self.config.max_steps_per_trial - 1:
                # Still allowed to wait — return same observation, no trial advancement
                self.steps_in_trial += 1
                obs = self._build_observation(
                    self.last_arm, self.last_signal, self.current_lights
                )
                info = self._build_info(phase_changed=False, forced=False)
                truncated = (
                    self.config.max_trials is not None
                    and self.total_trials >= self.config.max_trials
                )
                return obs, 0.0, False, truncated, info
            else:
                # Timeout — force random arm
                chosen_arm = int(self.np_random.integers(0, self.config.n_arms))
                forced = True
        else:
            # Take the first selected arm
            chosen_arm = int(selected_indices[0])
            forced = False

        # Evaluate the chosen arm against the current rule
        correct_arm = evaluate_rule(self.current_rule, self.current_lights)
        signal = int(chosen_arm == correct_arm)

        self.trial_in_phase += 1
        self.total_trials += 1
        self.steps_in_trial = 0

        # Check for phase transition
        phase_changed = False
        if self.trial_in_phase >= self.phase_duration:
            self.phase_id += 1
            self.trial_in_phase = 0
            self.current_rule = generate_random_rule(
                self.config.n_lights, self.config.n_arms, self.np_random
            )
            self.phase_duration = self._sample_phase_duration()
            phase_changed = True

        # Sample new lights for next trial
        self.current_lights = self._sample_light_config()
        self.last_arm = chosen_arm
        self.last_signal = signal

        obs = self._build_observation(self.last_arm, self.last_signal, self.current_lights)
        terminated = False
        truncated = (
            self.config.max_trials is not None
            and self.total_trials >= self.config.max_trials
        )
        info = self._build_info(phase_changed=phase_changed, forced=forced)

        return obs, 0.0, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            lines = [
                f"Phase: {self.phase_id} | Trial in phase: {self.trial_in_phase}/{self.phase_duration} | Total: {self.total_trials}",
                f"Lights: {list(self.current_lights)} | Last arm: {self.last_arm} | Signal: {self.last_signal}",
            ]
            return "\n".join(lines)
        return None

    def _sample_phase_duration(self) -> int:
        method = self.config.phase_change_method
        if method == "geometric":
            return int(self.np_random.geometric(self.config.phase_change_p))
        elif method == "uniform":
            lo, hi = self.config.phase_duration_range
            return int(self.np_random.integers(lo, hi + 1))
        elif method == "fixed":
            return self.config.fixed_phase_duration
        raise ValueError(f"Unknown phase_change_method: {method}")

    def _sample_light_config(self) -> tuple[int, ...]:
        bits = self.np_random.integers(0, 2, size=self.config.n_lights)
        return tuple(int(b) for b in bits)

    def _build_observation(
        self, selected_arm: int, arm_signal: int, lights: tuple[int, ...]
    ):
        if self.config.flatten_obs:
            # Shift selected_arm from [-1, K-1] to [0, K] for MultiDiscrete
            return np.array(
                [selected_arm + 1, arm_signal] + list(lights), dtype=np.int64
            )
        return {
            "selected_arm": np.int64(selected_arm),
            "arm_signal": np.int64(arm_signal),
            "lights": np.array(lights, dtype=np.int8),
        }

    def _build_info(self, phase_changed: bool, forced: bool) -> dict:
        correct_arm = evaluate_rule(self.current_rule, self.current_lights)
        return {
            "phase_id": self.phase_id,
            "trial_in_phase": self.trial_in_phase,
            "phase_changed": phase_changed,
            "correct_arm": correct_arm,
            "current_rule": dict(self.current_rule),
            "light_config": self.current_lights,
            "steps_in_trial": self.steps_in_trial,
            "forced": forced,
        }
