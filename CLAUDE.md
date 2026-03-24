# Three Pillars

## Project structure

```
environments/
  Conditional MAB/          # Gymnasium environment (installed as editable package)
    conditional_mab/
      config.py             # ConditionalMABConfig dataclass
      env.py                # ConditionalMABEnv (gym.Env)
      rule.py               # Rule generation & evaluation
    tests/
      test_env.py
agents/
  human/
    play.py                 # Interactive CLI agent for Conditional MAB
```

## Commands

- **Run tests:** `python3 -m pytest "environments/Conditional MAB/tests/" -v`
- **Human agent:** `python3 agents/human/play.py --n-lights 3 --n-arms 4 --seed 42`

## Python version

Python 3.11.9

## Conditional MAB environment

Non-stationary conditional deterministic multi-armed bandit. Binary lights provide context; agent selects an arm and observes a binary signal (1=correct, 0=wrong). The rule mapping lights→correct arm changes across phases.

### Key design decisions (Phase 2)

- **No reward**: reward is always 0.0. The agent must learn solely from `arm_signal` in the observation.
- **MultiBinary action space**: `MultiBinary(n_arms)` where all-zeros = "wait" (no-action). First `1`-index is the chosen arm. After `max_steps_per_trial` consecutive waits, the environment forces a random arm.
- **Info dict** includes `steps_in_trial` (int) and `forced` (bool) fields.
- Config field `max_steps_per_trial` (default 3, must be >= 1) controls deliberation budget.
