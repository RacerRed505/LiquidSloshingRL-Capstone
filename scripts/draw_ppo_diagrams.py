"""
PPO training flowchart and closed-loop control block diagram (second-review
items "ppo flowchart of process" and "+ block diagram / closed loop?").

Every number is read from config/simulation_config.py (TrainingConfig,
EnvironmentConfig) -- the same objects training/train_ppo.py and
environment/sloshing_env.py use -- so the diagrams cannot drift from the code.
The network size (7 -> 64 -> 64, tanh, separate actor and critic) is
stable-baselines3's MlpPolicy default, confirmed by loading models/ppo_sloshing.

    python -m scripts.draw_ppo_diagrams
    # writes figures/ppo_flowchart.{png,svg} and figures/control_block_diagram.{png,svg}
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from config.simulation_config import EnvironmentConfig, TrainingConfig  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#1b2733"
MUTED = "#6b7a88"
BLUE = "#2f6f93"
BLUE_SOFT = "#e3edf1"
GREEN = "#1e8449"
GREEN_SOFT = "#e4f2ec"
WARM = "#b9770e"
WARM_SOFT = "#f6ecdc"
RED = "#b23b3b"
RED_SOFT = "#f7e6e3"
GREY_SOFT = "#eef2f1"


def box(ax, cx, cy, w, h, title, body="", fc=BLUE_SOFT, ec=BLUE, title_size=11.5, body_size=9.5, tc=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
    if body:
        ax.text(cx, cy + h * 0.2, title, ha="center", va="center", fontsize=title_size,
                color=tc, fontweight="bold", zorder=3)
        ax.text(cx, cy - h * 0.17, body, ha="center", va="center", fontsize=body_size,
                color=tc, zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                color=tc, fontweight="bold", zorder=3)


def diamond(ax, cx, cy, w, h, text, fc=WARM_SOFT, ec=WARM):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10, color=INK, zorder=3, linespacing=1.3)


def arrow(ax, pts, color=INK, label=None, label_xy=None, ls="-", lw=1.6, label_color=None, fs=9.5):
    """Polyline arrow through pts, arrowhead at the last point."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text(*label_xy, label, fontsize=fs, color=label_color or color, ha="center", va="center",
                zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# ----------------------------------------------------------------------------
# PPO training flowchart
# ----------------------------------------------------------------------------

def draw_flowchart(tc: TrainingConfig, ec: EnvironmentConfig, out_stem: str) -> None:
    rollout = tc.n_steps * tc.n_envs
    n_updates = math.ceil(tc.total_timesteps / rollout)
    substeps = round(ec.control_dt / ec.physics_dt)
    ep_steps = round(ec.max_episode_seconds / ec.control_dt)
    lo, hi = ec.reset_disturbance_impulse_range

    fig = plt.figure(figsize=(13, 16.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.6, 13)
    ax.set_ylim(0, 16.5)
    ax.axis("off")
    ax.text(0.5, 16.0, "PPO training process", fontsize=19, color=INK, fontweight="bold")
    ax.text(0.5, 15.6, "training/train_ppo.py  +  environment/sloshing_env.py   (fast surrogate model)",
            fontsize=10.5, color=MUTED)

    X = 5.0
    W = 6.4
    box(ax, X, 14.75, W, 1.0, "1. Initialise networks",
        "Actor (policy) and critic (value): two separate MLPs, 7 → 64 → 64, tanh\n"
        f"Adam optimiser, learning rate {tc.learning_rate:g},  seed {tc.seed}")
    box(ax, X, 13.35, W, 1.0, f"2. Start {tc.n_envs} parallel environments",
        f"Each resets a level tank and kicks the liquid with a random impulse\n"
        f"of {lo:g}–{hi:g} N·s in a random direction")

    # Rollout loop (grouped).
    ax.add_patch(FancyBboxPatch((1.15, 5.95), 7.7, 6.65, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor=GREEN, lw=1.4, ls=(0, (5, 4)), zorder=0))
    ax.text(1.35, 12.35, f"3. Collect a rollout: {tc.n_steps} steps × {tc.n_envs} envs = {rollout:,} transitions",
            fontsize=11.5, color=GREEN, fontweight="bold")
    box(ax, X, 11.55, W, 0.85, "Observe state s  (7 values)",
        "θ, θ̇, fluid CoM x, y, vx, vy, slosh energy  →  normalised (VecNormalize)",
        fc=GREEN_SOFT, ec=GREEN)
    box(ax, X, 10.3, W, 0.85, "Actor picks action a ~ π(a | s)",
        f"Gaussian in [−1, 1]  →  motor torque τ = a × {ec.max_motor_torque_nm:g} N·m",
        fc=GREEN_SOFT, ec=GREEN)
    box(ax, X, 9.05, W, 0.85, f"Simulate {ec.control_dt * 1000:g} ms  ({substeps} physics steps of {ec.physics_dt * 1000:g} ms)",
        "Tank rotates under τ; Housner pendulum (the sloshing liquid) responds",
        fc=GREEN_SOFT, ec=GREEN)
    box(ax, X, 7.8, W, 0.85, "Reward r",
        f"r = −( {ec.slosh_energy_weight:g}·E_slosh + {ec.control_effort_weight:g}·τ² "
        f"+ {ec.angle_penalty_weight:g}·θ² )   (normalised)",
        fc=GREEN_SOFT, ec=GREEN)
    diamond(ax, X, 6.55, 3.6, 0.95, f"Episode over?\n|θ| > {math.degrees(ec.max_tank_angle_rad):.0f}°  or  {ec.max_episode_seconds:g} s")
    box(ax, 10.85, 6.55, 2.9, 0.8, "Reset that env", f"new random impulse\n(max {ep_steps} steps / episode)",
        fc=GREY_SOFT, ec=MUTED, title_size=10.5, body_size=9)

    for y0, y1 in [(14.25, 13.85), (12.85, 11.975), (11.125, 10.725), (9.875, 9.475), (8.625, 8.225), (7.375, 7.025)]:
        arrow(ax, [(X, y0), (X, y1)])
    arrow(ax, [(X + 1.8, 6.55), (9.4, 6.55)], label="yes", label_xy=(8.6, 6.75), color=INK)
    arrow(ax, [(10.85, 6.95), (10.85, 11.55), (X + W / 2, 11.55)], color=MUTED)
    # "no" loop back to observe (until the rollout buffer is full)
    arrow(ax, [(X - 1.8, 6.55), (0.75, 6.55), (0.75, 11.55), (X - W / 2, 11.55)], color=GREEN,
          label=f"no — next step\n(until {rollout:,} stored)", label_xy=(0.75, 9.1), fs=9)
    ax.text(10.85, 11.75, "", fontsize=9)

    # After the rollout.
    arrow(ax, [(X, 5.95), (X, 5.35)], label="buffer full", label_xy=(X + 0.85, 5.65), color=INK)
    box(ax, X, 4.85, W, 0.95, "4. Compute advantages (GAE)",
        f"Critic V(s) estimates future reward; γ = {tc.gamma:g},  λ = {tc.gae_lambda:g}\n"
        "advantage = how much better each action was than expected")
    arrow(ax, [(X, 4.375), (X, 3.95)])
    box(ax, X, 3.45, W, 0.95, "5. Update both networks",
        f"{tc.n_epochs} epochs over the rollout, minibatches of {tc.batch_size}\n"
        f"clipped objective (clip ε = {tc.clip_range:g}) + value loss  →  gradient step")
    arrow(ax, [(X, 2.975), (X, 2.6)])
    diamond(ax, X, 2.05, 4.0, 1.05, f"{tc.total_timesteps:,} steps done?\n(≈ {n_updates} updates)")
    arrow(ax, [(X + 2.0, 2.05), (9.4, 2.05)], label="yes", label_xy=(8.4, 2.25))
    box(ax, 10.85, 2.05, 2.9, 1.0, "6. Save", "policy + normalisation stats\nmodels/ppo_sloshing",
        fc=GREY_SOFT, ec=MUTED, title_size=10.5, body_size=9)
    arrow(ax, [(X - 2.0, 2.05), (0.2, 2.05), (0.2, 12.35), (1.15, 12.35)], color=BLUE,
          label="no — collect\nnext rollout", label_xy=(0.2, 4.0), fs=9)

    arrow(ax, [(10.85, 1.55), (10.85, 1.0)], color=MUTED)
    ax.text(10.85, 0.75, "then: evaluate (n=20), fine-tune on real\nChrono physics (50k steps), test on real SPH fluid",
            fontsize=9, color=MUTED, ha="center", va="center")

    # Side note on the right of the rollout block.
    ax.text(9.2, 15.25,
            "Why PPO?\n"
            "• works with a continuous\n  action (torque)\n"
            "• the clip ε keeps each\n  update small, so training\n  is stable\n"
            "• learns from experience;\n  needs no model of the fluid",
            fontsize=9.5, color=INK, va="top", linespacing=1.35,
            bbox=dict(facecolor=BLUE_SOFT, edgecolor=BLUE, boxstyle="round,pad=0.5"))

    _save(fig, out_stem)


# ----------------------------------------------------------------------------
# Closed-loop control block diagram
# ----------------------------------------------------------------------------

def draw_block_diagram(ec: EnvironmentConfig, out_stem: str) -> None:
    fig = plt.figure(figsize=(15, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(0.4, 7.1, "Closed-loop control: block diagram", fontsize=18, color=INK, fontweight="bold")
    ax.text(0.4, 6.72, f"Yes, it is closed loop: the controller measures the tank and liquid every "
                       f"{ec.control_dt * 1000:g} ms and feeds that back into its next torque command.",
            fontsize=10.5, color=MUTED)

    y = 4.4
    box(ax, 1.25, y, 1.9, 1.1, "Goal", "tank level (θ = 0)\nno sloshing", fc=GREY_SOFT, ec=MUTED,
        title_size=11, body_size=9.5)
    box(ax, 4.0, y, 2.6, 1.5, "Controller",
        "trained PPO policy\n(or PID / LQR baseline)\nstate → torque command", fc=BLUE_SOFT, ec=BLUE)
    box(ax, 6.95, y, 1.9, 1.1, "Torque limit",
        f"±{ec.max_motor_torque_nm:g} N·m (training)\n±7 N·m (real fluid)", fc=GREY_SOFT, ec=MUTED,
        title_size=10.5, body_size=8.8)
    box(ax, 9.25, y, 1.9, 1.1, "Motor", "torque-controlled\njoint at pivot O", fc=GREY_SOFT, ec=MUTED,
        title_size=11, body_size=9)

    # Plant: tank + liquid, coupled.
    ax.add_patch(FancyBboxPatch((10.55, 2.95), 4.05, 2.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor=RED, lw=1.5, zorder=0))
    ax.text(12.575, 5.62, "Plant (the physical system)", ha="center", fontsize=11, color=RED, fontweight="bold")
    box(ax, 12.575, 4.85, 3.6, 0.85, "Tank on single pivot", "rigid body, I_O; gravity tips it over",
        fc=RED_SOFT, ec=RED, title_size=10.5, body_size=9)
    box(ax, 12.575, 3.55, 3.6, 0.85, "Sloshing liquid", "Housner pendulum, or real SPH fluid",
        fc=RED_SOFT, ec=RED, title_size=10.5, body_size=9)
    ax.add_patch(FancyArrowPatch((12.2, 4.42), (12.2, 3.98), arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.4))
    ax.add_patch(FancyArrowPatch((12.95, 3.98), (12.95, 4.42), arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.4))
    ax.text(11.45, 4.2, "tank motion", fontsize=8.5, color=RED, ha="center")
    ax.text(13.75, 4.2, "fluid forces", fontsize=8.5, color=RED, ha="center")

    # Disturbance.
    ax.text(12.575, 6.45, "Disturbance d", ha="center", fontsize=10.5, color=WARM, fontweight="bold")
    ax.text(12.575, 6.17, "impulse on the liquid / road bumps", ha="center", fontsize=9, color=WARM)
    arrow(ax, [(12.575, 6.02), (12.575, 5.9)], color=WARM)

    # Forward path.
    arrow(ax, [(2.2, y), (2.7, y)])
    ax.text(2.45, y - 0.42, "via the\nreward", fontsize=8.5, color=MUTED, ha="center", va="center")
    arrow(ax, [(5.3, y), (6.0, y)], label="u", label_xy=(5.65, y + 0.22))
    arrow(ax, [(7.9, y), (8.3, y)], label="τ", label_xy=(8.1, y + 0.22))
    arrow(ax, [(10.2, y), (10.55, y)])

    # Feedback path.
    box(ax, 8.6, 1.35, 4.4, 1.15, "Sensors / state measurement",
        "θ, θ̇ (encoder / IMU) + liquid state:\nfluid CoM position & velocity, slosh energy",
        fc=GREEN_SOFT, ec=GREEN)
    arrow(ax, [(14.6, 3.6), (14.85, 3.6), (14.85, 1.35), (10.8, 1.35)], color=GREEN)
    ax.text(14.7, 2.4, "measured\noutputs\nθ, θ̇, liquid", fontsize=9.5, color=GREEN, ha="right", va="center")
    box(ax, 4.0, 1.35, 2.6, 0.95, "Normalise", "VecNormalize\n(running mean / std)", fc=GREEN_SOFT, ec=GREEN,
        title_size=10.5, body_size=9)
    arrow(ax, [(6.4, 1.35), (5.3, 1.35)], color=GREEN)
    arrow(ax, [(4.0, 1.825), (4.0, 3.65)], color=GREEN, label="state s (7 values)", label_xy=(4.0, 2.75))

    ax.text(0.4, 0.45,
            "Hardware-realistic variant (tested, on hold): the liquid-state sensors are replaced by two wall "
            "pressure taps + an IMU (environment/observable_sloshing_env.py).",
            fontsize=9.5, color=MUTED)

    # Training-only reward path.
    ax.add_patch(FancyBboxPatch((0.4, 2.55), 2.3, 1.05, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="white", edgecolor=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2))
    ax.text(1.55, 3.25, "Training only", ha="center", fontsize=10, color=MUTED, fontweight="bold", zorder=3)
    ax.text(1.55, 2.86, "reward r → PPO\nupdates the controller", ha="center", fontsize=8.8, color=MUTED,
            zorder=3, linespacing=1.3)
    arrow(ax, [(2.7, 3.08), (3.2, 3.08), (3.2, 3.65)], color=MUTED, ls=(0, (4, 3)), lw=1.2)

    _save(fig, out_stem)


def draw_block_diagram_vertical_bw(ec: EnvironmentConfig, out_stem: str) -> None:
    """Same block diagram, laid out top-to-bottom in black and white (for print).

    Same content as draw_block_diagram: forward path down the centre column,
    feedback path up the right column, training-only reward path on the left.
    """
    k, w = "black", "white"
    fig = plt.figure(figsize=(11, 14))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.text(5.5, 13.45, "Closed-loop control: block diagram", ha="center", fontsize=18, color=k, fontweight="bold")
    ax.text(5.5, 12.95, f"The controller measures the tank and liquid every {ec.control_dt * 1000:g} ms\n"
                        "and feeds that back into its next torque command.",
            ha="center", va="center", fontsize=10.5, color=k)

    cx, rx = 4.2, 8.6  # forward-path column, feedback column
    box(ax, cx, 11.75, 3.0, 1.0, "Goal", "tank level (θ = 0)\nno sloshing", fc=w, ec=k, tc=k,
        title_size=11, body_size=9.5)
    box(ax, cx, 10.0, 3.2, 1.4, "Controller",
        "trained PPO policy\n(or PID / LQR baseline)\nstate → torque command", fc=w, ec=k, tc=k)
    box(ax, cx, 8.3, 3.0, 1.0, "Torque limit",
        f"±{ec.max_motor_torque_nm:g} N·m (training)\n±7 N·m (real fluid)", fc=w, ec=k, tc=k,
        title_size=10.5, body_size=8.8)
    box(ax, cx, 6.75, 3.0, 1.0, "Motor", "torque-controlled\njoint at pivot O", fc=w, ec=k, tc=k,
        title_size=11, body_size=9)

    # Plant: tank + liquid, coupled.
    ax.add_patch(FancyBboxPatch((2.2, 2.6), 4.0, 3.15, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor=k, lw=1.5, zorder=0))
    ax.text(cx, 5.42, "Plant (the physical system)", ha="center", fontsize=11, color=k, fontweight="bold")
    box(ax, cx, 4.6, 3.6, 0.85, "Tank on single pivot", "rigid body, I_O; gravity tips it over",
        fc=w, ec=k, tc=k, title_size=10.5, body_size=9)
    box(ax, cx, 3.2, 3.6, 0.85, "Sloshing liquid", "Housner pendulum, or real SPH fluid",
        fc=w, ec=k, tc=k, title_size=10.5, body_size=9)
    ax.add_patch(FancyArrowPatch((3.8, 4.17), (3.8, 3.63), arrowstyle="-|>", mutation_scale=12, color=k, lw=1.4))
    ax.add_patch(FancyArrowPatch((4.6, 3.63), (4.6, 4.17), arrowstyle="-|>", mutation_scale=12, color=k, lw=1.4))
    ax.text(3.7, 3.9, "tank motion", fontsize=8.5, color=k, ha="right", va="center")
    ax.text(4.7, 3.9, "fluid forces", fontsize=8.5, color=k, ha="left", va="center")

    # Disturbance, from below.
    ax.text(cx, 1.75, "Disturbance d", ha="center", fontsize=10.5, color=k, fontweight="bold")
    ax.text(cx, 1.45, "impulse on the liquid / road bumps", ha="center", fontsize=9, color=k)
    arrow(ax, [(cx, 2.0), (cx, 2.6)], color=k)

    # Forward path, top to bottom.
    arrow(ax, [(cx, 11.25), (cx, 10.7)], color=k)
    ax.text(cx + 0.15, 10.98, "via the reward", fontsize=8.5, color=k, ha="left", va="center")
    arrow(ax, [(cx, 9.3), (cx, 8.8)], label="u", label_xy=(cx + 0.3, 9.05), color=k)
    arrow(ax, [(cx, 7.8), (cx, 7.25)], label="τ", label_xy=(cx + 0.3, 7.52), color=k)
    arrow(ax, [(cx, 6.25), (cx, 5.75)], color=k)

    # Feedback path, bottom to top.
    box(ax, rx, 4.0, 3.4, 1.45, "Sensors / state\nmeasurement",
        "θ, dθ/dt (encoder / IMU)\n+ liquid state: fluid CoM\nposition & velocity, slosh energy", fc=w, ec=k, tc=k,
        title_size=10.5, body_size=8.8)
    arrow(ax, [(6.2, 4.0), (6.9, 4.0)], color=k)
    ax.text(6.55, 3.55, "measured\noutputs", fontsize=8.5, color=k, ha="center", va="center")
    box(ax, rx, 7.5, 3.0, 1.0, "Normalise", "VecNormalize\n(running mean / std)", fc=w, ec=k, tc=k,
        title_size=10.5, body_size=9)
    arrow(ax, [(rx, 4.725), (rx, 7.0)], color=k)
    arrow(ax, [(rx, 8.0), (rx, 10.0), (5.8, 10.0)], label="state s\n(7 values)", label_xy=(rx, 9.1), color=k)

    # Training-only reward path, on the left.
    ax.add_patch(FancyBboxPatch((0.3, 9.45), 1.9, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=w, edgecolor=k, lw=1.2, ls=(0, (4, 3)), zorder=2))
    ax.text(1.25, 10.25, "Training only", ha="center", fontsize=10, color=k, fontweight="bold", zorder=3)
    ax.text(1.25, 9.83, "reward r → PPO\nupdates the\ncontroller", ha="center", va="center", fontsize=8.5,
            color=k, zorder=3, linespacing=1.25)
    arrow(ax, [(2.2, 10.0), (2.6, 10.0)], ls=(0, (4, 3)), lw=1.2, color=k)

    ax.text(5.5, 0.55,
            "Hardware-realistic variant (tested, on hold): the liquid-state sensors are replaced\n"
            "by two wall pressure taps + an IMU (environment/observable_sloshing_env.py).",
            ha="center", va="center", fontsize=9.5, color=k)

    _save(fig, out_stem)


def _save(fig, out_stem: str) -> None:
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_stem}.png and .svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "figures"))
    args = parser.parse_args()
    tc, ec = TrainingConfig(), EnvironmentConfig()
    draw_flowchart(tc, ec, os.path.join(args.out_dir, "ppo_flowchart"))
    draw_block_diagram(ec, os.path.join(args.out_dir, "control_block_diagram"))
    draw_block_diagram_vertical_bw(ec, os.path.join(args.out_dir, "control_block_diagram_vertical_bw"))


if __name__ == "__main__":
    main()
