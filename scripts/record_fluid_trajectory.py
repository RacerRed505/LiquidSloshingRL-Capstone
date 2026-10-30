"""
Phase 13-14 -- Record a local (PyChrono-FSI-free) rollout's tank_motion
and PendulumBackend's own predicted fluid state, for later comparison
against physics/sph_backend.py::SPHBackend on Colab.

Per the design decision in the development log Entry 41, SPHBackend cannot
run on this project's local machine (no CUDA GPU) and is not meant to
run inside RL training or even a live local environment -- the bridge
is trajectory-file-based instead: this script runs entirely against the
fast surrogate (TankDynamicsModel + PendulumBackend, both PyChrono-free)
to produce a CSV of (tank_angle, tank_angular_velocity) per step -- the
exact tank_motion sequence SPHBackend.step() needs to replay -- plus
PendulumBackend's own predicted fluid_com/slosh_energy for the same
steps, so scripts/compare_fluid_backends.py has something to diff
SPHBackend's later Colab-produced output against.

Uses the default SloshingSuppressionEnv construction (fast surrogate
tank + PendulumBackend) directly -- no ChronoTankDynamics, no PyChrono
import anywhere in this script, so it runs on any machine with just the
pure-Python dependencies (matches TrainingConfig's own n_envs docstring:
"cheap, pure-Python, no pychrono").

Run:
    python -m scripts.record_fluid_trajectory --seed 0
    python -m scripts.record_fluid_trajectory models/ppo_sloshing --seed 0 --output tank_motion_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from typing import Optional

import gymnasium as gym
import numpy as np

import environment  # noqa: F401 -- import for its gym.register() side effect

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENV_ID = "LiquidSloshingRL/SloshingSuppression-v0"


def record(
    model_path: Optional[str],
    seed: int,
    deterministic: bool = True,
) -> list:
    """Roll out one episode, recording tank_motion + PendulumBackend's own
    fluid_com/slosh_energy prediction at every step. Returns a list of
    row dicts, one per step."""
    env = gym.make(ENV_ID)

    model = None
    if model_path is not None:
        from stable_baselines3 import PPO

        model = PPO.load(model_path)
        logger.info("Using trained policy from %s", model_path)
    else:
        logger.info("No model given -- using random actions (same as scripts/random_policy_demo.py).")

    obs, info = env.reset(seed=seed)
    rows = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": step,
                "tank_angle": info["tank_angle"],
                "tank_angular_velocity": info["tank_angular_velocity"],
                "pendulum_fluid_com_x": float(obs[2]),
                "pendulum_fluid_com_y": float(obs[3]),
                "pendulum_fluid_com_vx": float(obs[4]),
                "pendulum_fluid_com_vy": float(obs[5]),
                "pendulum_slosh_energy": info["slosh_energy"],
                "pendulum_slosh_potential_energy": info["slosh_potential_energy"],
                "pendulum_slosh_total_energy": info["slosh_total_energy"],
            }
        )
        step += 1

    env.close()
    logger.info("Recorded %d steps (seed=%d).", len(rows), seed)
    return rows


def write_csv(path: str, rows: list) -> None:
    if not rows:
        raise ValueError("No rows to write -- the episode terminated immediately.")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote tank_motion + pendulum-prediction CSV to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path", nargs="?", default=None,
        help="Optional saved PPO model path. Omit to use random actions instead.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of the deterministic policy mean (only relevant with model_path).")
    parser.add_argument("--output", default="tank_motion_trajectory.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = record(args.model_path, args.seed, deterministic=not args.stochastic)
    write_csv(args.output, rows)
    logger.info(
        "Next: upload %s to the shared Colab Drive folder and run SPHBackend "
        "against its tank_angle/tank_angular_velocity columns, then compare "
        "with scripts/compare_fluid_backends.py.",
        args.output,
    )
