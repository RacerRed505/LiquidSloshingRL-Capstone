"""
Black-and-white, top-to-bottom flowcharts of the project's simulation and
control code, for the research paper's "Appendix A: Soft Code Flowcharts".

Each chart follows the real code path it is named after (file names are
printed on the chart), and every number is read from
config/simulation_config.py so the charts cannot drift from the code.
Standard flowchart symbols: rounded terminator (start/end), rectangle
(process), parallelogram (input/output), diamond (decision).

    python -m scripts.draw_code_flowcharts
    # writes figures/appendix_flowcharts/A1_...A5_*.{png,svg}

The vertical black-and-white control block diagram (A6) comes from
scripts/draw_ppo_diagrams.py::draw_block_diagram_vertical_bw.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from config.simulation_config import EnvironmentConfig, TrainingConfig  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = "black"
FS = 9.5  # body text size


class Chart:
    """A portrait figure in inch-like data units, black on white."""

    def __init__(self, height: float, title: str, subtitle: str, width: float = 8.5):
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.w, self.h = width, height
        self.ax.text(width / 2, height - 0.45, title, ha="center", va="center", fontsize=15, fontweight="bold")
        self.ax.text(width / 2, height - 0.85, subtitle, ha="center", va="center", fontsize=9.5,
                     style="italic")

    def _text(self, cx, cy, text, size=FS):
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=K, zorder=3, linespacing=1.3)

    def term(self, cx, cy, text, w=2.4, h=0.55):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                         facecolor="white", edgecolor=K, lw=1.4, zorder=2))
        self._text(cx, cy, text, 10)
        return (cx, cy, w, h)

    def proc(self, cx, cy, text, w=5.0, h=0.75):
        self.ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white", edgecolor=K, lw=1.4, zorder=2))
        self._text(cx, cy, text)
        return (cx, cy, w, h)

    def io(self, cx, cy, text, w=5.0, h=0.75):
        s = 0.25
        pts = [(cx - w / 2 + s, cy + h / 2), (cx + w / 2 + s, cy + h / 2),
               (cx + w / 2 - s, cy - h / 2), (cx - w / 2 - s, cy - h / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=K, lw=1.4, zorder=2))
        self._text(cx, cy, text)
        return (cx, cy, w, h)

    def dec(self, cx, cy, text, w=3.6, h=1.1):
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=K, lw=1.4, zorder=2))
        self._text(cx, cy, text, 9)
        return (cx, cy, w, h)

    def arrow(self, pts, label=None, label_xy=None, ha="center"):
        for p, q in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([p[0], q[0]], [p[1], q[1]], color=K, lw=1.3, zorder=1)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=13,
                                          color=K, lw=1.3, shrinkA=0, shrinkB=0, zorder=1))
        if label:
            self.ax.text(*label_xy, label, fontsize=9, ha=ha, va="center", zorder=4,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    def down(self, a, b, label=None):
        """Straight arrow from the bottom of node a to the top of node b."""
        self.arrow([(a[0], a[1] - a[3] / 2), (b[0], b[1] + b[3] / 2)], label,
                   (a[0] + 0.15, (a[1] - a[3] / 2 + b[1] + b[3] / 2) / 2), ha="left")

    def note(self, text):
        self.ax.text(self.w / 2, 0.35, text, ha="center", va="center", fontsize=8.5, style="italic")

    def save(self, out_stem):
        os.makedirs(os.path.dirname(out_stem), exist_ok=True)
        for ext in ("png", "svg"):
            self.fig.savefig(f"{out_stem}.{ext}", dpi=170, facecolor="white")
        plt.close(self.fig)
        print(f"Saved {out_stem}.png and .svg")


def right_of(n):
    return (n[0] + n[2] / 2, n[1])


def left_of(n):
    return (n[0] - n[2] / 2, n[1])


# ----------------------------------------------------------------------------
# A.1 Overall program flow
# ----------------------------------------------------------------------------

def a1_overall(tc: TrainingConfig, out: str) -> None:
    c = Chart(14.0, "A.1  Overall program flow",
              "Order in which the project's programs were run, from model building to results")
    x = 4.25
    steps = [
        ("term", "Start", 12.6),
        ("proc", "Build the tank model (builders/)\ntank walls, pivot and motor from primitives;\nmass and inertia computed analytically", 11.55),
        ("proc", f"Train PPO on the fast surrogate (training/train_ppo.py)\n{tc.total_timesteps:,} steps, {tc.n_envs} parallel environments — see A.3", 10.4),
        ("proc", "Evaluate the policy (evaluation/evaluate_policy.py)\n20 episodes on the surrogate", 9.35),
        ("proc", "Fine-tune on rigid-body Chrono physics\n(training/train_ppo_real_physics.py, 50,000 steps)", 8.3),
        ("io", "Record a tank-angle trajectory to CSV\n(scripts/record_fluid_trajectory.py)", 7.25),
        ("proc", "Replay it in the SPH solver at dx = 10, 5, 3, 2 mm\n(scripts/run_sph_backend_trajectory.py) — see A.4", 6.2),
        ("proc", "Compare SPH with the pendulum and hydrostatic models\n(scripts/compare_fluid_backends.py, make_cfd_figures.py)", 5.15),
        ("proc", "Fine-tune and test in the two-way coupled SPH tank\n(scripts/supervised_closed_loop_finetune.py,\nrun_closed_loop_sph_policy.py) — see A.5", 3.95),
        ("io", "Write results: CSV files, figures, report and paper", 2.75),
        ("term", "End", 1.75),
    ]
    nodes = []
    for kind, text, y in steps:
        if kind == "term":
            nodes.append(c.term(x, y, text))
        elif kind == "io":
            nodes.append(c.io(x, y, text, w=5.6, h=0.75))
        else:
            nodes.append(c.proc(x, y, text, w=6.2, h=0.9 if text.count("\n") >= 2 else 0.75))
    for a, b in zip(nodes, nodes[1:]):
        c.down(a, b)
    c.note("Steps that need Chrono::FSI (A.4, A.5) ran on the RTX 3090 and RTX 5060 machines.")
    c.save(out)


# ----------------------------------------------------------------------------
# A.2 Control code: one episode of the environment
# ----------------------------------------------------------------------------

def a2_episode(ec: EnvironmentConfig, out: str) -> None:
    subs = ec.substeps_per_control_step
    lo, hi = ec.reset_disturbance_impulse_range
    c = Chart(15.0, "A.2  Control code: one episode",
              "environment/sloshing_env.py — reset() and step(); the controller is PPO, PID or LQR")
    x = 4.25
    start = c.term(x, 13.6, "Start episode")
    reset = c.proc(x, 12.55, f"reset(): reset tank and fluid model;\napply a random impulse of {lo:g}–{hi:g} N·s to the liquid", w=6.0)
    obs = c.io(x, 11.4, "Build observation s (7 values):\nθ, dθ/dt, fluid CoM x, y, vx, vy, slosh energy E", w=5.6)
    ctl = c.proc(x, 10.25, "Controller computes action a ∈ [−1, 1]\n(PPO policy, or PID / LQR baseline)", w=5.6)
    tau = c.proc(x, 9.2, f"Clip a to [−1, 1];  torque τ = a × {ec.max_motor_torque_nm:g} N·m", w=5.6, h=0.6)
    sub = c.proc(x, 8.1, f"Physics substep ({ec.physics_dt * 1000:g} ms):\nadvance tank under τ, then advance fluid\nmodel with the new tank angle and rate", w=5.6, h=0.95)
    dsub = c.dec(x, 6.75, f"{subs} substeps\ndone?")
    rew = c.proc(x, 5.45, f"Read fluid state;  reward\nr = −({ec.slosh_energy_weight:g}·E + {ec.control_effort_weight:g}·τ² + {ec.angle_penalty_weight:g}·θ²)", w=5.6)
    dang = c.dec(x, 4.15, f"|θ| > {math.degrees(ec.max_tank_angle_rad):.0f}°?")
    dlen = c.dec(x, 2.75, f"{ec.max_episode_steps} steps\nreached?")
    end = c.term(x, 1.35, "End episode")

    for a, b in [(start, reset), (reset, obs), (obs, ctl), (ctl, tau), (tau, sub), (sub, dsub)]:
        c.down(a, b)
    c.down(dsub, rew, "yes")
    c.down(rew, dang)
    c.down(dang, dlen, "no")
    # substep loop back (right side)
    c.arrow([right_of(dsub), (7.6, 6.75), (7.6, 8.1), right_of(sub)], "no", (7.6, 7.4))
    # tilt too far -> end (right side)
    c.arrow([right_of(dang), (7.6, 4.15), (7.6, 1.35), right_of(end)], "yes — terminated", (7.6, 3.4))
    c.arrow([(dlen[0], dlen[1] - dlen[3] / 2), (end[0], end[1] + end[3] / 2)], "yes — truncated",
            (x + 0.15, 2.05), ha="left")
    # next control step (left side)
    c.arrow([left_of(dlen), (0.55, 2.75), (0.55, 11.4), (x - 2.8 - 0.25, 11.4)], "no — next\ncontrol step",
            (0.55, 7.0))
    c.note(f"One control step = {ec.control_dt * 1000:g} ms (50 Hz); the same loop is used for training (A.3) and evaluation.")
    c.save(out)


# ----------------------------------------------------------------------------
# A.3 PPO training
# ----------------------------------------------------------------------------

def a3_training(tc: TrainingConfig, ec: EnvironmentConfig, out: str) -> None:
    rollout = tc.n_steps * tc.n_envs
    n_updates = math.ceil(tc.total_timesteps / rollout)
    c = Chart(13.0, "A.3  Control code: PPO training",
              "training/train_ppo.py (stable-baselines3 PPO, MlpPolicy)")
    x = 4.25
    start = c.term(x, 11.6, "Start")
    init = c.proc(x, 10.6, f"Initialise actor and critic networks (7 → 64 → 64, tanh)\nAdam optimiser, learning rate {tc.learning_rate:g}, seed {tc.seed}", w=6.4)
    envs = c.proc(x, 9.5, f"Create {tc.n_envs} parallel environments (A.2)\nwrapped in VecNormalize (observation and reward scaling)", w=6.4)
    roll = c.proc(x, 8.35, f"Collect a rollout: {tc.n_steps} steps × {tc.n_envs} environments\n= {rollout:,} transitions (s, a, r), resetting episodes as they end", w=6.4)
    gae = c.proc(x, 7.2, f"Compute advantages with GAE\n(γ = {tc.gamma:g}, λ = {tc.gae_lambda:g}) using the critic's value estimates", w=6.4)
    upd = c.proc(x, 6.05, f"Update both networks: {tc.n_epochs} epochs, minibatches of {tc.batch_size},\nclipped objective (ε = {tc.clip_range:g}) + value loss", w=6.4)
    ck = c.io(x, 4.9, f"Every {tc.checkpoint_every_steps:,} steps: save a checkpoint", w=5.6, h=0.6)
    done = c.dec(x, 3.7, f"{tc.total_timesteps:,}\nsteps done?", w=3.4)
    save = c.io(x, 2.45, "Save policy + normalisation statistics\n(models/ppo_sloshing)", w=5.6)
    end = c.term(x, 1.4, "End")
    for a, b in [(start, init), (init, envs), (envs, roll), (roll, gae), (gae, upd), (upd, ck), (ck, done)]:
        c.down(a, b)
    c.down(done, save, "yes")
    c.down(save, end)
    c.arrow([left_of(done), (0.45, 3.7), (0.45, 8.35), (x - 3.2, 8.35)], "no — next\nrollout", (0.45, 6.0))
    c.note(f"About {n_updates} update rounds in total.")
    c.save(out)


# ----------------------------------------------------------------------------
# A.4 SPH simulation (one-way, prescribed tank motion)
# ----------------------------------------------------------------------------

def a4_sph(ec: EnvironmentConfig, out: str) -> None:
    c = Chart(15.0, "A.4  Simulation code: SPH replay of a tank trajectory",
              "scripts/run_sph_backend_trajectory.py + physics/sph_backend.py (Chrono::FSI, one-way coupling)")
    x = 4.25
    start = c.term(x, 13.6, "Start")
    read = c.io(x, 12.6, "Read the recorded tank-motion CSV\n(angle θᵢ and rate (dθ/dt)ᵢ every 20 ms)", w=5.6)
    skip = c.dec(x, 11.4, "Output file already\ncomplete?", w=3.8)
    build = c.proc(x, 10.0, "Build the SPH scene: tank walls → boundary (BCE) markers;\nwater block on a lattice of spacing dx; WCSPH settings\n(Tait EOS, δ-SPH, XSPH, artificial viscosity, ADAMI walls)", w=6.6, h=1.0)
    row = c.proc(x, 8.75, "Take the next row i: target θᵢ, (dθ/dt)ᵢ", w=5.0, h=0.55)
    sub = c.proc(x, 7.75, "Substep Δt: move the tank angle by linear interpolation\ntowards θᵢ, then DoStepDynamics(Δt)", w=5.6)
    dsub = c.dec(x, 6.5, "0.02 s covered?", w=3.4, h=0.9)
    rec = c.proc(x, 5.35, "Read particles: liquid centroid (tank frame), velocity,\nbulk kinetic energy; append a row", w=6.2)
    sav = c.io(x, 4.25, "Every 25 rows: save the CSV (and a particle snapshot)", w=5.8, h=0.6)
    dall = c.dec(x, 3.15, "All rows done?", w=3.2, h=0.9)
    fin = c.io(x, 2.1, "Write the final CSV; log run time", w=5.0, h=0.6)
    end = c.term(x, 1.2, "End")
    for a, b in [(start, read), (read, skip)]:
        c.down(a, b)
    c.down(skip, build, "no")
    for a, b in [(build, row), (row, sub), (sub, dsub)]:
        c.down(a, b)
    c.down(dsub, rec, "yes")
    c.down(rec, sav)
    c.down(sav, dall)
    c.down(dall, fin, "yes")
    c.down(fin, end)
    c.arrow([right_of(skip), (7.9, 11.4), (7.9, 1.2), right_of(end)], "yes — skip\n(unless --force)", (7.9, 6.6))
    c.arrow([left_of(dsub), (1.0, 6.5), (1.0, 7.75), (x - 2.8, 7.75)], "no", (1.0, 7.12))
    c.arrow([left_of(dall), (0.5, 3.15), (0.5, 8.75), (x - 2.5, 8.75)], "no — next row", (0.5, 6.0))
    c.note("Δt = 1×10⁻⁴, 5×10⁻⁵, 3×10⁻⁵, 2×10⁻⁵ s at dx = 10, 5, 3, 2 mm; the solver may shorten it further (variable step).")
    c.save(out)


# ----------------------------------------------------------------------------
# A.5 Closed-loop SPH control (two-way coupling)
# ----------------------------------------------------------------------------

def a5_closed_loop(ec: EnvironmentConfig, out: str) -> None:
    c = Chart(15.5, "A.5  Control + simulation code: closed loop in SPH",
              "scripts/run_closed_loop_sph_policy.py + physics/closed_loop_sph_tank.py (two-way coupling)")
    x = 4.25
    start = c.term(x, 14.1, "Start")
    load = c.io(x, 13.1, "Load the trained policy and its normalisation statistics", w=5.8, h=0.6)
    build = c.proc(x, 12.05, "Build the two-way SPH tank: torque motor at the pivot,\nwater on a lattice, ADAMI walls; reset with a 0.05 rad/s kick", w=6.4)
    obs = c.proc(x, 10.9, "Build observation s from θ, dθ/dt and the fluid state", w=5.4, h=0.6)
    pre = c.dec(x, 9.85, "|dθ/dt| > 1.0 rad/s?", w=3.4, h=0.9)
    act = c.proc(x, 8.75, "Policy predicts a;  τ = clip(a) × torque limit (7 N·m)\nset motor torque τ", w=6.0)
    sub = c.proc(x, 7.65, "Substep Δt: DoStepDynamics(Δt)\n(fluid forces act back on the tank)", w=5.4)
    trip = c.dec(x, 6.5, "|dθ/dt| > 1.0 rad/s?", w=3.4, h=0.9)
    dsub = c.dec(x, 5.3, "0.02 s covered?", w=3.4, h=0.9)
    rec = c.io(x, 4.2, "Record θ, dθ/dt, τ, liquid centroid and energy", w=5.4, h=0.6)
    dn = c.dec(x, 3.1, "All steps done?", w=3.2, h=0.9)
    save = c.io(x, 2.05, "Save the CSV and particle snapshots", w=5.0, h=0.6)
    end = c.term(x, 1.15, "End")
    for a, b in [(start, load), (load, build), (build, obs), (obs, pre)]:
        c.down(a, b)
    c.down(pre, act, "no")
    c.down(act, sub)
    c.down(sub, trip)
    c.down(trip, dsub, "no")
    c.down(dsub, rec, "yes")
    c.down(rec, dn)
    c.down(dn, save, "yes")
    c.down(save, end)
    c.arrow([left_of(dsub), (1.25, 5.3), (1.25, 7.65), (x - 2.7, 7.65)], "no", (1.25, 6.5))
    c.arrow([right_of(pre), (8.0, 9.85), (8.0, 2.05), (x + 2.75, 2.05)], "yes — safety\nstop", (8.0, 9.2))
    c.ax.plot([right_of(trip)[0], 8.0], [6.5, 6.5], color=K, lw=1.3, zorder=1)
    c.ax.plot([8.0], [6.5], marker="o", color=K, markersize=4, zorder=2)
    c.ax.text(7.0, 6.75, "yes — stop\nmid-step", fontsize=9, ha="center", va="center")
    c.arrow([left_of(dn), (0.5, 3.1), (0.5, 10.9), (x - 2.7, 10.9)], "no — next\ncontrol step", (0.5, 7.0))
    c.note("The 1.0 rad/s check runs before every control step and after every substep, so the run stops before the solver diverges.")
    c.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "figures", "appendix_flowcharts"))
    args = parser.parse_args()
    tc, ec = TrainingConfig(), EnvironmentConfig()
    d = args.out_dir
    a1_overall(tc, os.path.join(d, "A1_overall_program_flow"))
    a2_episode(ec, os.path.join(d, "A2_control_episode"))
    a3_training(tc, ec, os.path.join(d, "A3_ppo_training"))
    a4_sph(ec, os.path.join(d, "A4_sph_simulation"))
    a5_closed_loop(ec, os.path.join(d, "A5_closed_loop_sph"))
    from scripts.draw_ppo_diagrams import draw_block_diagram_vertical_bw
    draw_block_diagram_vertical_bw(ec, os.path.join(d, "A6_control_block_diagram"))


if __name__ == "__main__":
    main()
