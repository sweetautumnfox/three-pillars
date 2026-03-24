from dataclasses import dataclass, field


@dataclass
class ConditionalMABConfig:
    """Configuration for the Conditional MAB environment."""

    n_lights: int = 3
    n_arms: int = 4
    phase_change_method: str = "geometric"  # "geometric" | "uniform" | "fixed"
    phase_change_p: float = 0.05  # geometric parameter
    phase_duration_range: tuple[int, int] = (20, 60)  # uniform range
    fixed_phase_duration: int = 40
    max_steps_per_trial: int = 3
    max_trials: int | None = None
    flatten_obs: bool = False

    def validate(self) -> None:
        if self.n_lights < 1:
            raise ValueError(f"n_lights must be >= 1, got {self.n_lights}")
        if self.n_lights > 20:
            raise ValueError(
                f"n_lights must be <= 20 (2^{self.n_lights} configs), got {self.n_lights}"
            )
        if self.n_arms < 2:
            raise ValueError(f"n_arms must be >= 2, got {self.n_arms}")
        if self.phase_change_method not in ("geometric", "uniform", "fixed"):
            raise ValueError(
                f"phase_change_method must be 'geometric', 'uniform', or 'fixed', "
                f"got '{self.phase_change_method}'"
            )
        if self.phase_change_method == "geometric":
            if not (0 < self.phase_change_p <= 1):
                raise ValueError(
                    f"phase_change_p must be in (0, 1], got {self.phase_change_p}"
                )
        if self.phase_change_method == "uniform":
            lo, hi = self.phase_duration_range
            if lo < 1 or hi < lo:
                raise ValueError(
                    f"phase_duration_range must satisfy 1 <= lo <= hi, "
                    f"got ({lo}, {hi})"
                )
        if self.phase_change_method == "fixed":
            if self.fixed_phase_duration < 1:
                raise ValueError(
                    f"fixed_phase_duration must be >= 1, got {self.fixed_phase_duration}"
                )
        if self.max_steps_per_trial < 1:
            raise ValueError(
                f"max_steps_per_trial must be >= 1, got {self.max_steps_per_trial}"
            )
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError(f"max_trials must be >= 1 or None, got {self.max_trials}")
