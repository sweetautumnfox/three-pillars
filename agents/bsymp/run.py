#!/usr/bin/env python3
"""Run BSyMP agent on the Conditional MAB environment.

Examples
--------
# Basic run, static phase, more states:
python3 agents/bsymp/run.py --phase-method fixed --fixed-phase-duration 500 \\
    --n-extra-states 4 --max-trials 500 --seed 42

# Realtime plot:
python3 agents/bsymp/run.py --plot --max-trials 300 --seed 42

# Neurogenesis with plot:
python3 agents/bsymp/run.py --plot --neurogenesis --neurogenesis-p 0.05 \\
    --max-trials 300 --seed 42

# Tune priors:
python3 agents/bsymp/run.py --a-prior 0.5 --c-A-init 0.5 --session-length 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo / "environments" / "Conditional MAB"))
sys.path.insert(0, str(_repo / "agents"))

from conditional_mab import ConditionalMABConfig, ConditionalMABEnv

from bsymp.config import BSyMPConfig
from bsymp.agent import BSyMPAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BSyMP agent on Conditional MAB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Environment ---
    env = p.add_argument_group("environment")
    env.add_argument("--n-lights", type=int, default=3)
    env.add_argument("--n-arms", type=int, default=4)
    env.add_argument("--max-trials", type=int, default=500)
    env.add_argument("--phase-method", type=str, default="geometric",
                     choices=["geometric", "uniform", "fixed"])
    env.add_argument("--phase-p", type=float, default=0.05,
                     help="geometric p (expected phase len = 1/p)")
    env.add_argument("--fixed-phase-duration", type=int, default=40,
                     help="phase length when --phase-method=fixed")
    env.add_argument("--phase-lo", type=int, default=20,
                     help="uniform range low")
    env.add_argument("--phase-hi", type=int, default=60,
                     help="uniform range high")

    # --- Model ---
    mdl = p.add_argument_group("model")
    mdl.add_argument("--n-extra-states", type=int, default=0,
                     help="extra hidden states beyond n_arms")
    mdl.add_argument("--a-prior", type=float, default=1.0,
                     help="Dirichlet concentration for A (likelihood)")
    mdl.add_argument("--b-prior", type=float, default=1.0,
                     help="Dirichlet concentration for B (transition)")
    mdl.add_argument("--c-A-init", type=float, default=0.9,
                     help="initial connectivity for C_A")
    mdl.add_argument("--c-B-init", type=float, default=0.9,
                     help="initial connectivity for C_B")
    mdl.add_argument("--session-length", type=int, default=10,
                     help="timesteps between connectivity updates")

    # --- Neurogenesis ---
    ngen = p.add_argument_group("neurogenesis")
    ngen.add_argument("--neurogenesis", action="store_true", default=False)
    ngen.add_argument("--neurogenesis-p", type=float, default=0.02)
    ngen.add_argument("--neurogenesis-max-states", type=int, default=20)

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument("--seed", type=int, default=42)
    out.add_argument("--log-interval", type=int, default=10)
    out.add_argument("--plot", action="store_true", default=False,
                     help="realtime matplotlib dashboard")
    out.add_argument("--plot-interval", type=int, default=1,
                     help="update plot every N trials")

    return p.parse_args()


# ======================================================================
# Realtime plotting
# ======================================================================

class LiveDashboard:
    """Five-panel realtime matplotlib dashboard.

    Layout (2 rows, 3 columns via gridspec):
        Top:     accuracy | action posteriors | network graph
        Bottom:  C_A heatmap | C_B heatmap
    """

    def __init__(self, n_arms: int, n_lights: int, window: int = 50):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyArrowPatch
        self.plt = plt
        self.FancyArrowPatch = FancyArrowPatch

        self.n_arms = n_arms
        self.n_lights = n_lights
        self.window = window

        self.trials: list[int] = []
        self.overall_acc: list[float] = []
        self.window_acc: list[float] = []
        self.correct_buf: list[int] = []

        self.action_posterior_history: list[np.ndarray] = []
        self.action_trial_indices: list[int] = []

        # Store latest observation for node coloring
        self.last_o_t: np.ndarray | None = None

        self.fig = plt.figure(figsize=(17, 10))
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               height_ratios=[1, 1], hspace=0.35, wspace=0.3)
        self.fig.suptitle("BSyMP Dashboard", fontsize=14)

        # Panel 1: accuracy (top-left)
        self.ax_acc = self.fig.add_subplot(gs[0, 0])
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Trial")
        self.ax_acc.set_ylabel("Accuracy")
        self.line_overall, = self.ax_acc.plot([], [], "b-", label="overall")
        self.line_window, = self.ax_acc.plot([], [], "r-", alpha=0.6,
                                             label=f"window({window})")
        self.ax_acc.legend(loc="upper left", fontsize=8)

        # Panel 2: action posteriors (top-center)
        self.ax_act = self.fig.add_subplot(gs[0, 1])
        self.ax_act.set_title("Action-node posteriors")
        self.ax_act.set_xlabel("Trial")
        self.ax_act.set_ylabel("Posterior")
        self.act_lines = []
        for arm in range(n_arms):
            ln, = self.ax_act.plot([], [], label=f"arm {arm}", alpha=0.7)
            self.act_lines.append(ln)
        self.ax_act.legend(loc="upper left", fontsize=8)

        # Panel 3: network graph (top-right)
        self.ax_net = self.fig.add_subplot(gs[0, 2])
        self.ax_net.set_title("Network")
        self.ax_net.set_aspect("equal")

        # Panel 4: C_A heatmap (bottom-left)
        self.ax_cA = self.fig.add_subplot(gs[1, 0])
        self.ax_cA.set_title("C_A (obs → states)")

        # Panel 5: C_B heatmap (bottom-center+right, spanning 2 cols)
        self.ax_cB = self.fig.add_subplot(gs[1, 1:])
        self.ax_cB.set_title("C_B (states → states)")

        plt.ion()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record_trial(self, trial: int, correct: int,
                     action_posterior: np.ndarray,
                     o_t: np.ndarray) -> None:
        self.correct_buf.append(correct)
        total_correct = sum(self.correct_buf)
        self.trials.append(trial)
        self.overall_acc.append(total_correct / trial)

        buf = self.correct_buf[-self.window:]
        self.window_acc.append(sum(buf) / len(buf))

        self.action_posterior_history.append(action_posterior.copy())
        self.action_trial_indices.append(trial)
        self.last_o_t = o_t.copy()

    # ------------------------------------------------------------------
    # Network graph drawing
    # ------------------------------------------------------------------

    def _draw_network(self, model) -> None:
        """Draw observation and hidden-state nodes with connectivity edges."""
        ax = self.ax_net
        ax.clear()
        ax.set_title("Network")
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        N_o = model.N_o
        N_s = model.N_s
        n_arms = self.n_arms

        # --- Node positions ---
        # Obs nodes: left column (x=0)
        obs_positions = []
        for i in range(N_o):
            y = 1.0 - i / max(N_o - 1, 1)
            obs_positions.append((0.0, y))

        # State nodes: right column (x=1), action nodes on top
        state_positions = []
        for j in range(N_s):
            y = 1.0 - j / max(N_s - 1, 1)
            state_positions.append((1.0, y))

        # --- Draw C_A edges (obs -> state) ---
        for i in range(N_o):
            for j in range(N_s):
                w = float(model.c_A[i, j])
                if w < 0.05:
                    continue
                x0, y0 = obs_positions[i]
                x1, y1 = state_positions[j]
                color = self.plt.cm.RdYlGn(w)
                ax.plot([x0, x1], [y0, y1], color=color,
                        linewidth=w * 2.5, alpha=max(w, 0.15), zorder=1)

        # --- Draw C_B edges (state -> state, as curved self-loops / arcs) ---
        for i in range(N_s):
            for j in range(N_s):
                w = float(model.c_B[i, j])
                if w < 0.05:
                    continue
                x0, y0 = state_positions[j]  # from j at t-1
                x1, y1 = state_positions[i]  # to i at t
                if i == j:
                    # Self-loop: small arc to the right
                    arc = self.FancyArrowPatch(
                        (x0 + 0.06, y0 + 0.02), (x0 + 0.06, y0 - 0.02),
                        connectionstyle="arc3,rad=-0.8",
                        color=self.plt.cm.cool(w),
                        linewidth=w * 2, alpha=max(w, 0.2),
                        arrowstyle="->,head_length=3,head_width=2",
                        zorder=1,
                    )
                    ax.add_patch(arc)
                else:
                    # Curved arrow between different state nodes
                    arc = self.FancyArrowPatch(
                        (x0 + 0.04, y0), (x1 + 0.04, y1),
                        connectionstyle="arc3,rad=0.3",
                        color=self.plt.cm.cool(w),
                        linewidth=w * 2, alpha=max(w, 0.2),
                        arrowstyle="->,head_length=3,head_width=2",
                        zorder=1,
                    )
                    ax.add_patch(arc)

        # --- Draw observation nodes ---
        obs_labels = [f"L{i}" for i in range(self.n_lights)] + ["sig"]
        for i, (x, y) in enumerate(obs_positions):
            # Color by current observation value
            val = float(self.last_o_t[i]) if self.last_o_t is not None else 0.5
            fc = self.plt.cm.Oranges(val * 0.6 + 0.2)
            ax.plot(x, y, "s", markersize=18, color=fc,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=3)
            ax.text(x, y, obs_labels[i], ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=4)

        # --- Draw hidden state nodes ---
        for j, (x, y) in enumerate(state_positions):
            val = float(model.s[j])
            if j < n_arms:
                # Action node
                fc = self.plt.cm.Blues(val * 0.7 + 0.2)
                marker = "o"
                label = f"A{j}"
            else:
                # Internal node
                fc = self.plt.cm.Purples(val * 0.7 + 0.2)
                marker = "o"
                label = f"I{j - n_arms}"
            ax.plot(x, y, marker, markersize=18, color=fc,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=3)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=4)
            # Posterior value below
            ax.text(x, y - 0.06, f"{val:.2f}", ha="center", va="top",
                    fontsize=6, color="gray", zorder=4)

        # --- Legend ---
        ax.text(0.0, -0.07, "obs", ha="center", fontsize=8, color="gray")
        ax.text(1.0, -0.07, "states", ha="center", fontsize=8, color="gray")

    # ------------------------------------------------------------------

    def update(self, model) -> None:
        """Redraw all panels."""
        if not self.trials:
            return

        # Panel 1: accuracy
        self.line_overall.set_data(self.trials, self.overall_acc)
        self.line_window.set_data(self.trials, self.window_acc)
        self.ax_acc.set_xlim(0, max(self.trials[-1], 1))
        self.ax_acc.set_ylim(0, max(max(self.overall_acc + self.window_acc) + 0.05, 0.5))

        # Panel 2: action posteriors
        arr = np.array(self.action_posterior_history)  # (T, n_arms)
        xs = self.action_trial_indices
        for arm in range(min(self.n_arms, arr.shape[1])):
            self.act_lines[arm].set_data(xs, arr[:, arm])
        self.ax_act.set_xlim(0, max(xs[-1], 1))
        self.ax_act.set_ylim(-0.05, 1.05)

        # Panel 3: network graph
        self._draw_network(model)

        # Panel 4: C_A heatmap
        self.ax_cA.clear()
        self.ax_cA.set_title("C_A (obs → states)")
        self.ax_cA.imshow(model.c_A, aspect="auto", vmin=0, vmax=1,
                          cmap="RdYlGn")
        self.ax_cA.set_xlabel("Hidden state")
        self.ax_cA.set_ylabel("Observation")
        for i in range(model.c_A.shape[0]):
            for j in range(model.c_A.shape[1]):
                self.ax_cA.text(j, i, f"{model.c_A[i,j]:.2f}",
                                ha="center", va="center", fontsize=7)

        # Panel 5: C_B heatmap
        self.ax_cB.clear()
        self.ax_cB.set_title("C_B (states → states)")
        self.ax_cB.imshow(model.c_B, aspect="auto", vmin=0, vmax=1,
                          cmap="RdYlGn")
        self.ax_cB.set_xlabel("State (t-1)")
        self.ax_cB.set_ylabel("State (t)")
        for i in range(model.c_B.shape[0]):
            for j in range(model.c_B.shape[1]):
                self.ax_cB.text(j, i, f"{model.c_B[i,j]:.2f}",
                                ha="center", va="center", fontsize=7)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ======================================================================
# Main loop
# ======================================================================

def main() -> None:
    args = parse_args()

    env_config = ConditionalMABConfig(
        n_lights=args.n_lights,
        n_arms=args.n_arms,
        max_steps_per_trial=3,
        phase_change_method=args.phase_method,
        phase_change_p=args.phase_p,
        fixed_phase_duration=args.fixed_phase_duration,
        phase_duration_range=(args.phase_lo, args.phase_hi),
        max_trials=args.max_trials,
    )
    env = ConditionalMABEnv(config=env_config)

    agent_config = BSyMPConfig(
        n_lights=args.n_lights,
        n_arms=args.n_arms,
        session_length=args.session_length,
        n_extra_states=args.n_extra_states,
        a_prior=args.a_prior,
        b_prior=args.b_prior,
        c_A_init=args.c_A_init,
        c_B_init=args.c_B_init,
        neurogenesis_enabled=args.neurogenesis,
        neurogenesis_p=args.neurogenesis_p,
        neurogenesis_max_states=args.neurogenesis_max_states,
        max_trials=args.max_trials,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    agent = BSyMPAgent(agent_config)

    dashboard: LiveDashboard | None = None
    if args.plot:
        dashboard = LiveDashboard(args.n_arms, args.n_lights)

    obs, info = env.reset(seed=args.seed)

    trial_count = 0
    correct_count = 0
    window_correct = 0
    window_size = 0

    print(f"BSyMP Agent | N_o={agent.N_o}, N_s={agent.model.N_s}")
    print(
        f"Lights={args.n_lights}, Arms={args.n_arms}, "
        f"Session={args.session_length}, Neurogenesis={args.neurogenesis}"
    )
    print(
        f"Priors: a={args.a_prior}, b={args.b_prior}, "
        f"c_A={args.c_A_init}, c_B={args.c_B_init}"
    )
    print(
        f"Phase: method={args.phase_method}, "
        + (f"p={args.phase_p}" if args.phase_method == "geometric"
           else f"dur={args.fixed_phase_duration}" if args.phase_method == "fixed"
           else f"range=({args.phase_lo},{args.phase_hi})")
    )
    print("-" * 60)

    done = False
    while not done:
        action = agent.step(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get("forced") or np.any(action):
            trial_count += 1
            is_correct = int(obs["arm_signal"])
            correct_count += is_correct
            window_correct += is_correct
            window_size += 1

            if dashboard is not None:
                o_t = agent.encode_observation(obs)
                dashboard.record_trial(
                    trial_count, is_correct,
                    agent.model.s[: args.n_arms],
                    o_t,
                )
                if trial_count % args.plot_interval == 0:
                    dashboard.update(agent.model)

            if window_size >= args.log_interval:
                accuracy = window_correct / window_size
                overall = correct_count / trial_count
                print(
                    f"Trial {trial_count:4d} | "
                    f"Phase {info['phase_id']:2d} | "
                    f"N_s={agent.model.N_s:2d} | "
                    f"Win acc={accuracy:.2f} | "
                    f"Overall={overall:.3f}"
                )
                window_correct = 0
                window_size = 0

    print("-" * 60)
    overall = correct_count / max(trial_count, 1)
    print(
        f"Done: {trial_count} trials, "
        f"{correct_count}/{trial_count} correct ({overall:.3f})"
    )
    print(f"Final N_s: {agent.model.N_s}")
    chance = 1.0 / args.n_arms
    print(f"Chance level: {chance:.3f}")

    if dashboard is not None:
        dashboard.update(agent.model)
        print("\nClose the plot window to exit.")
        dashboard.plt.ioff()
        dashboard.plt.show()


if __name__ == "__main__":
    main()
