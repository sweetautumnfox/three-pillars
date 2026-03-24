#!/usr/bin/env python3
"""Interactive human agent for the Conditional MAB environment.

Run:
    python agents/human/play.py --n-lights 3 --n-arms 4 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "environments" / "Conditional MAB"))

from conditional_mab import ConditionalMABConfig, ConditionalMABEnv


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def fmt_lights(lights: tuple[int, ...]) -> str:
    return "".join("●" if b else "○" for b in lights)


def fmt_lights_spaced(lights: tuple[int, ...]) -> list[str]:
    return ["●" if b else "○" for b in lights]


def render_state(
    env: ConditionalMABEnv,
    info: dict,
    history: deque,
    steps_in_trial: int,
):
    cfg = env.config
    n_lights = cfg.n_lights
    n_arms = cfg.n_arms
    max_steps = cfg.max_steps_per_trial

    phase_id = info["phase_id"]
    rule = info["current_rule"]

    clear_screen()

    W = 50  # box width

    def hline(left="├", right="┤", fill="─"):
        return left + fill * (W - 2) + right

    lines: list[str] = []
    lines.append("┌" + "─" * (W - 2) + "┐")

    # Title
    title = f"CONDITIONAL MAB    Phase {phase_id}   Trial {env.total_trials}"
    lines.append(f"│  {title:<{W - 4}}│")
    step_line = f"Step {steps_in_trial + 1}/{max_steps} in current trial"
    lines.append(f"│  {step_line:<{W - 4}}│")

    # Lights
    lines.append(hline())
    light_hdr = "│ LIGHTS │"
    for i in range(n_lights):
        light_hdr += f" L{i:<2}│"
    lines.append(f"{light_hdr:<{W - 1}}│")

    light_vals = "│        │"
    for b in env.current_lights:
        sym = " ● " if b else " ○ "
        light_vals += f"{sym}│"
    lines.append(f"{light_vals:<{W - 1}}│")

    # Arms header
    lines.append(hline())
    arm_hdr = "│ ARMS   │"
    for i in range(n_arms):
        arm_hdr += f" A{i:<2}│"
    lines.append(f"{arm_hdr:<{W - 1}}│")
    lines.append(hline())

    # Last action result
    if env.last_arm >= 0:
        sig_sym = "●" if env.last_signal else "○"
        last_line = f"Last: arm {env.last_arm} → {sig_sym} (signal {env.last_signal})"
    else:
        last_line = "Last: (none)"
    lines.append(f"│  {last_line:<{W - 4}}│")

    # Rule table
    lines.append(hline())
    lines.append(f"│  {'RULE TABLE (Phase ' + str(phase_id) + ')':<{W - 4}}│")

    # Header for rule table
    light_cols = " ".join(f"L{i}" for i in range(n_lights))
    rule_hdr = f"  {light_cols} │ Arm"
    lines.append(f"│  {rule_hdr:<{W - 4}}│")

    # Sort rule keys for consistent display
    for config_key in sorted(rule.keys()):
        bits = " ".join(f"{b:2d}" for b in config_key)
        arm_val = rule[config_key]
        row = f"  {bits} │  {arm_val}"
        lines.append(f"│  {row:<{W - 4}}│")

    # History
    lines.append(hline())
    hist_title = f"HISTORY (last {min(5, len(history))})"
    lines.append(f"│  {hist_title:<{W - 4}}│")
    for entry in reversed(history):
        trial_num, lights_t, arm, signal = entry
        sig_sym = "●" if signal else "○"
        row = f"  #{trial_num} [{fmt_lights(lights_t)}] → A{arm} → {sig_sym}"
        lines.append(f"│  {row:<{W - 4}}│")

    lines.append("└" + "─" * (W - 2) + "┘")

    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play the Conditional MAB as a human")
    p.add_argument("--n-lights", type=int, default=3)
    p.add_argument("--n-arms", type=int, default=4)
    p.add_argument("--max-steps-per-trial", type=int, default=3)
    p.add_argument("--phase-method", type=str, default="geometric",
                   choices=["geometric", "uniform", "fixed"])
    p.add_argument("--phase-p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    config = ConditionalMABConfig(
        n_lights=args.n_lights,
        n_arms=args.n_arms,
        max_steps_per_trial=args.max_steps_per_trial,
        phase_change_method=args.phase_method,
        phase_change_p=args.phase_p,
    )
    env = ConditionalMABEnv(config=config)
    obs, info = env.reset(seed=args.seed)

    history: deque = deque(maxlen=5)
    steps_in_trial = 0

    render_state(env, info, history, steps_in_trial)

    while True:
        prompt = f"Enter arm [0-{config.n_arms - 1}], Enter=wait, q=quit: "
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        raw = raw.strip()
        if raw.lower() == "q":
            print("Goodbye!")
            break

        if raw == "":
            # No-action (wait)
            action = np.zeros(config.n_arms, dtype=np.int8)
        else:
            try:
                arm = int(raw)
            except ValueError:
                print(f"Invalid input. Enter 0-{config.n_arms - 1}, Enter, or q.")
                continue
            if not (0 <= arm < config.n_arms):
                print(f"Arm must be 0-{config.n_arms - 1}.")
                continue
            action = np.zeros(config.n_arms, dtype=np.int8)
            action[arm] = 1

        # Remember lights before step (for history)
        lights_before = env.current_lights

        obs, reward, terminated, truncated, info = env.step(action)
        steps_in_trial = info["steps_in_trial"]

        # Record history only when an arm was actually selected
        if info.get("forced") or np.any(action):
            selected = obs["selected_arm"]
            signal = obs["arm_signal"]
            history.append((env.total_trials, lights_before, int(selected), int(signal)))

        render_state(env, info, history, steps_in_trial)

        if info.get("forced"):
            print(f"  ⏱ Timeout! Environment forced arm {obs['selected_arm']}.")

        if terminated or truncated:
            print("Episode ended.")
            break


if __name__ == "__main__":
    main()
