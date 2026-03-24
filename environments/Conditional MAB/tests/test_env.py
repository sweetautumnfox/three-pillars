import sys
from pathlib import Path

import numpy as np
import pytest

# Add the package to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conditional_mab import ConditionalMABConfig, ConditionalMABEnv


def _one_hot(arm: int, n_arms: int) -> np.ndarray:
    """Helper: create a one-hot MultiBinary action."""
    a = np.zeros(n_arms, dtype=np.int8)
    a[arm] = 1
    return a


def _no_action(n_arms: int) -> np.ndarray:
    """Helper: all-zeros (wait) action."""
    return np.zeros(n_arms, dtype=np.int8)


class TestConfig:
    def test_default_valid(self):
        config = ConditionalMABConfig()
        config.validate()

    def test_invalid_n_lights(self):
        with pytest.raises(ValueError, match="n_lights"):
            ConditionalMABConfig(n_lights=0).validate()

    def test_n_lights_too_large(self):
        with pytest.raises(ValueError, match="n_lights"):
            ConditionalMABConfig(n_lights=21).validate()

    def test_invalid_n_arms(self):
        with pytest.raises(ValueError, match="n_arms"):
            ConditionalMABConfig(n_arms=1).validate()

    def test_invalid_phase_change_p(self):
        with pytest.raises(ValueError, match="phase_change_p"):
            ConditionalMABConfig(phase_change_p=0.0).validate()

    def test_invalid_phase_duration_range(self):
        with pytest.raises(ValueError, match="phase_duration_range"):
            ConditionalMABConfig(
                phase_change_method="uniform", phase_duration_range=(0, 10)
            ).validate()

    def test_invalid_fixed_duration(self):
        with pytest.raises(ValueError, match="fixed_phase_duration"):
            ConditionalMABConfig(
                phase_change_method="fixed", fixed_phase_duration=0
            ).validate()

    def test_invalid_max_steps_per_trial(self):
        with pytest.raises(ValueError, match="max_steps_per_trial"):
            ConditionalMABConfig(max_steps_per_trial=0).validate()


class TestEnvironmentContract:
    def setup_method(self):
        self.env = ConditionalMABEnv(n_lights=3, n_arms=4)

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=42)
        assert self.env.observation_space.contains(obs)

    def test_step_returns_valid_tuple(self):
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(_one_hot(0, 4))
        assert self.env.observation_space.contains(obs)
        assert reward == 0.0
        assert terminated is False
        assert isinstance(truncated, bool)

    def test_reset_sentinel_values(self):
        obs, _ = self.env.reset(seed=42)
        assert obs["selected_arm"] == -1
        assert obs["arm_signal"] == 0

    def test_info_keys(self):
        _, info = self.env.reset(seed=42)
        for key in ("phase_id", "trial_in_phase", "phase_changed", "correct_arm",
                     "current_rule", "steps_in_trial", "forced"):
            assert key in info

    def test_step_updates_last_arm(self):
        self.env.reset(seed=42)
        obs, _, _, _, _ = self.env.step(_one_hot(2, 4))
        assert obs["selected_arm"] == 2


class TestSignalCorrectness:
    def test_correct_arm_gives_signal_1(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=3)
        obs, info = env.reset(seed=42)
        correct = info["correct_arm"]
        obs, reward, _, _, _ = env.step(_one_hot(correct, 3))
        assert reward == 0.0
        assert obs["arm_signal"] == 1

    def test_wrong_arm_gives_signal_0(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=3)
        obs, info = env.reset(seed=42)
        correct = info["correct_arm"]
        wrong = (correct + 1) % 3
        obs, reward, _, _, _ = env.step(_one_hot(wrong, 3))
        assert reward == 0.0
        assert obs["arm_signal"] == 0


class TestRewardAlwaysZero:
    def test_reward_always_zero(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=3)
        env.reset(seed=42)
        for _ in range(20):
            arm = int(env.np_random.integers(0, 3))
            _, reward, _, _, _ = env.step(_one_hot(arm, 3))
            assert reward == 0.0


class TestNoAction:
    def test_no_action_keeps_lights_same(self):
        env = ConditionalMABEnv(n_lights=3, n_arms=4, max_steps_per_trial=3)
        obs, _ = env.reset(seed=42)
        lights_before = obs["lights"].copy()

        obs, reward, _, _, info = env.step(_no_action(4))
        assert reward == 0.0
        np.testing.assert_array_equal(obs["lights"], lights_before)
        assert info["steps_in_trial"] == 1
        assert info["forced"] is False

    def test_no_action_preserves_last_arm_signal(self):
        env = ConditionalMABEnv(n_lights=3, n_arms=4, max_steps_per_trial=3)
        obs, _ = env.reset(seed=42)
        arm_before = obs["selected_arm"]
        signal_before = obs["arm_signal"]

        obs, _, _, _, _ = env.step(_no_action(4))
        assert obs["selected_arm"] == arm_before
        assert obs["arm_signal"] == signal_before

    def test_timeout_forces_random_selection(self):
        env = ConditionalMABEnv(n_lights=3, n_arms=4, max_steps_per_trial=3)
        env.reset(seed=42)

        # Wait twice (steps 0→1, 1→2), then third no-action hits timeout
        env.step(_no_action(4))
        env.step(_no_action(4))
        obs, reward, _, _, info = env.step(_no_action(4))

        assert reward == 0.0
        assert info["forced"] is True
        # After forced selection, steps_in_trial resets
        assert info["steps_in_trial"] == 0
        # An arm was selected
        assert obs["selected_arm"] >= 0

    def test_max_steps_per_trial_1_forces_immediately(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=3, max_steps_per_trial=1)
        env.reset(seed=42)

        # With max_steps_per_trial=1, any no-action immediately forces
        obs, _, _, _, info = env.step(_no_action(3))
        assert info["forced"] is True
        assert obs["selected_arm"] >= 0


class TestOneHotAction:
    def test_first_selected_arm_is_used(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=4)
        env.reset(seed=42)
        # Action with multiple bits set — first index should be used
        action = np.array([0, 1, 0, 1], dtype=np.int8)
        obs, _, _, _, info = env.step(action)
        assert obs["selected_arm"] == 1
        assert info["forced"] is False

    def test_valid_one_hot_selects_correct_arm(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=3)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(_one_hot(2, 3))
        assert obs["selected_arm"] == 2


class TestDeterministicSeeding:
    def test_same_seed_same_sequence(self):
        env1 = ConditionalMABEnv(n_lights=3, n_arms=4)
        env2 = ConditionalMABEnv(n_lights=3, n_arms=4)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        for key in obs1:
            np.testing.assert_array_equal(obs1[key], obs2[key])

        for _ in range(50):
            action = _one_hot(0, 4)
            r1 = env1.step(action)
            r2 = env2.step(action)
            # Compare rewards
            assert r1[1] == r2[1]
            # Compare observations
            for key in r1[0]:
                np.testing.assert_array_equal(r1[0][key], r2[0][key])


class TestPhaseTransitions:
    def test_phase_eventually_changes(self):
        env = ConditionalMABEnv(
            n_lights=2, n_arms=2, phase_change_method="fixed", fixed_phase_duration=5
        )
        env.reset(seed=42)

        for _ in range(20):
            _, _, _, _, info = env.step(_one_hot(0, 2))

        assert info["phase_id"] >= 1

    def test_phase_changed_flag(self):
        env = ConditionalMABEnv(
            n_lights=2, n_arms=2, phase_change_method="fixed", fixed_phase_duration=3
        )
        env.reset(seed=42)

        phase_change_seen = False
        for _ in range(10):
            _, _, _, _, info = env.step(_one_hot(0, 2))
            if info["phase_changed"]:
                phase_change_seen = True
                break

        assert phase_change_seen


class TestTruncation:
    def test_truncation_at_max_trials(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=2, max_trials=10)
        env.reset(seed=42)

        for i in range(9):
            _, _, _, truncated, _ = env.step(_one_hot(0, 2))
            assert not truncated

        _, _, _, truncated, _ = env.step(_one_hot(0, 2))
        assert truncated

    def test_no_truncation_without_max_trials(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=2)
        env.reset(seed=42)

        for _ in range(100):
            _, _, _, truncated, _ = env.step(_one_hot(0, 2))
            assert not truncated


class TestFlattenedObs:
    def test_flattened_obs_space(self):
        env = ConditionalMABEnv(n_lights=3, n_arms=4, flatten_obs=True)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert obs.shape == (5,)  # arm + signal + 3 lights

    def test_flattened_step(self):
        env = ConditionalMABEnv(n_lights=3, n_arms=4, flatten_obs=True)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(_one_hot(1, 4))
        assert env.observation_space.contains(obs)


class TestRender:
    def test_ansi_render(self):
        env = ConditionalMABEnv(n_lights=2, n_arms=2, render_mode="ansi")
        env.reset(seed=42)
        output = env.render()
        assert isinstance(output, str)
        assert "Phase:" in output
