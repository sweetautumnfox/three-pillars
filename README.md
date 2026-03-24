# Three Pillars

An experimental framework exploring three jointly necessary conditions for intelligence:

1. **Active Inference** -- inference, belief updating, and action governed by free energy minimization.
2. **Environmental Forcing** -- environments that prevent stable equilibria and demand continuous learning.
3. **Architectural Headroom** -- endogenous structural reorganization driven by the same variational dynamics.

The repo pairs a non-stationary bandit environment (Pillar 2) with agents that embody Pillars 1 and 3.

## Project structure

```
environments/
  Conditional MAB/          # Gymnasium environment (editable package)
    conditional_mab/
      config.py             # ConditionalMABConfig dataclass
      env.py                # ConditionalMABEnv (gym.Env)
      rule.py               # Rule generation & evaluation
    tests/
      test_env.py
agents/
  human/
    play.py                 # Interactive CLI agent
  bsymp/
    agent.py                # BSyMPAgent
    model.py                # BSyMPModel (variational inference)
    neurogenesis.py          # Neurogenesis strategies (Pillar 3)
    config.py               # BSyMPConfig dataclass
    run.py                  # CLI runner with optional live dashboard
```

## Setup

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the Conditional MAB environment as an editable package
pip install -e "environments/Conditional MAB"

# 3. Install agent dependencies
pip install scipy matplotlib
```

## Running the human agent

Play the Conditional MAB interactively in your terminal. Each step you choose an arm (or wait) and observe a binary signal indicating whether you picked the correct arm for the current light configuration.

```bash
python3 agents/human/play.py --n-lights 3 --n-arms 4 --seed 42
```

Controls: type an arm number (`0`--`3`), press Enter to wait, or `q` to quit.

| Flag | Default | Description |
|------|---------|-------------|
| `--n-lights` | 3 | Number of binary context lights |
| `--n-arms` | 4 | Number of arms to choose from |
| `--max-steps-per-trial` | 3 | Wait budget before the env forces a random arm |
| `--phase-method` | `geometric` | Phase-change schedule (`geometric`, `uniform`, `fixed`) |
| `--phase-p` | 0.05 | Geometric phase-change probability |
| `--seed` | None | Random seed |

## Running the BSyMP agent

BSyMP (Bayesian Synaptic Message Passing) is a variational-inference agent that learns the lights-to-arm mapping online and adapts when the rule changes.

```bash
# Basic run
python3 agents/bsymp/run.py --max-trials 500 --seed 42

# With realtime matplotlib dashboard
python3 agents/bsymp/run.py --plot --max-trials 300 --seed 42

# Static phase (longer, fixed rule duration)
python3 agents/bsymp/run.py --phase-method fixed --fixed-phase-duration 500 \
    --n-extra-states 4 --max-trials 500 --seed 42

# Enable neurogenesis (Pillar 3: architectural headroom)
python3 agents/bsymp/run.py --plot --neurogenesis --neurogenesis-p 0.05 \
    --max-trials 300 --seed 42

# Tune priors
python3 agents/bsymp/run.py --a-prior 0.5 --c-A-init 0.5 --session-length 20
```

### BSyMP flags

| Flag | Default | Description |
|------|---------|-------------|
| **Environment** | | |
| `--n-lights` | 3 | Number of binary context lights |
| `--n-arms` | 4 | Number of arms |
| `--max-trials` | 500 | Total trials to run |
| `--phase-method` | `geometric` | Phase schedule (`geometric`, `uniform`, `fixed`) |
| `--phase-p` | 0.05 | Geometric phase-change probability |
| `--fixed-phase-duration` | 40 | Phase length when method is `fixed` |
| `--phase-lo` / `--phase-hi` | 20 / 60 | Uniform phase duration range |
| **Model** | | |
| `--n-extra-states` | 0 | Extra hidden states beyond n_arms |
| `--a-prior` | 1.0 | Dirichlet concentration for likelihood (A) |
| `--b-prior` | 1.0 | Dirichlet concentration for transitions (B) |
| `--c-A-init` | 0.9 | Initial observation-to-state connectivity |
| `--c-B-init` | 0.9 | Initial state-to-state connectivity |
| `--session-length` | 10 | Steps between connectivity updates |
| **Neurogenesis** | | |
| `--neurogenesis` | off | Enable neurogenesis |
| `--neurogenesis-p` | 0.02 | Per-step probability of adding a state |
| `--neurogenesis-max-states` | 20 | Upper bound on hidden states |
| **Output** | | |
| `--seed` | 42 | Random seed |
| `--log-interval` | 10 | Print accuracy every N trials |
| `--plot` | off | Show realtime matplotlib dashboard |
| `--plot-interval` | 1 | Update plot every N trials |

## Running tests

```bash
python3 -m pytest "environments/Conditional MAB/tests/" -v
```
