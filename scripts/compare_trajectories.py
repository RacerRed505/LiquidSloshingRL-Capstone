"""
Phase 11 -- Compare a single-episode trajectory between the analytic
surrogate tank (environment/tank_dynamics.py) and the real ChSystem tank
(environment/chrono_tank_dynamics.py), under the SAME trained policy and
the SAME reset seed.

scripts/validate_real_dynamics.py (Phase 10) already answered "is there a
gap" (yes: mean return -0.0008 surrogate vs -0.0093 real, ~11.6x, on 3
episodes -- see CONTINUE_HERE.md). This script exists to answer WHERE
that gap comes from: same policy, same starting fluid disturbance
(because both envs are reset with the same seed and
SloshingSuppressionEnv.reset() draws its impulse from self.np_random,
seeded identically -- see that method), two physics backends -- so the
two trajectories start identical and diverge purely because the tank
itself responds differently step by step, which then feeds back into
what the policy sees and does next.

Deliberately does NOT use SB3's VecEnv/DummyVecEnv machinery the way
evaluate_policy.py does, even though that means re-implementing a small
slice of its VecNormalize handling here. Reason: evaluate_policy.py's
own VecNormalize branch never passes a seed into vec_env.reset() (see
its own code), so two separate DummyVecEnv-wrapped episodes are NOT
guaranteed to start from the same fluid disturbance -- fine for
aggregate statistics over many episodes, but wrong for this script's
whole purpose (a fair single-episode comparison). Driving each raw env
directly with reset(seed=seed) and normalizing observations by hand
via a loaded VecNormalize object's .normalize_obs() (which only reads
its stored obs_rms, no live env needed) keeps the seeding under this
script's control while still feeding the policy exactly the normalized
observations it was trained on.

Produces:
    <output_prefix>.csv -- per-step values for both runs, one row per
        (source, step)
    <output_prefix>.png -- angle and cumulative-reward comparison plots
        (requires matplotlib; skipped with a logged warning, CSV still
        written, if it's not installed -- not in requirements.txt, see
        that file)

Run:
    python -m scripts.compare_trajectories models/ppo_sloshing --seed 0
    python -m scripts.compare_trajectories models/ppo_sloshing --seed 3 --output-prefix trajectory_seed3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import environment  # noqa: F401 -- import for its gym.register() side effect
from config.simulation_config import EnvironmentConfig, PivotConfig, TankConfig, WorldConfig
from environment.sloshing_env import SloshingSuppressionEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENV_ID = "LiquidSloshingRL/SloshingSuppression-v0"


@dataclass
class Trajectory:
    """Per-step recording for one rollout. Lists, not arrays -- episode
    length isn't known up front (an episode can terminate early if
    max_tank_angle_rad is exceeded, not just truncate at
    max_episode_steps -- see SloshingSuppressionEnv.step())."""

    label: str
    tank_angle: List[float] = field(default_factory=list)
    tank_angular_velocity: List[float] = field(default_factory=list)
    slosh_energy: List[float] = field(default_factory=list)
    commanded_torque_nm: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    cumulative_reward: List[float] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return self.cumulative_reward[-1] if self.cumulative_reward else 0.0


def _build_real_dynamics_env() -> SloshingSuppressionEnv:
    """Same construction as scripts/validate_real_dynamics.py's own
    helper -- deliberately duplicated rather than imported, since that
    module is a `__main__`-oriented script, not a library one; importing
    a private-by-convention helper across scripts would couple two
    independent entry points together for no real benefit here."""
    from environment.chrono_tank_dynamics import ChronoTankDynamics

    tank_config = TankConfig(fixed=False)
    pivot_config = PivotConfig()
    world_config = WorldConfig()
    env_config = EnvironmentConfig()
    tank_dynamics = ChronoTankDynamics(
        tank_config=tank_config,
        pivot_config=pivot_config,
        world_config=world_config,
        env_config=env_config,
    )
    return SloshingSuppressionEnv(
        tank_config=tank_config,
        pivot_config=pivot_config,
        world_config=world_config,
        env_config=env_config,
        tank_dynamics=tank_dynamics,
    )


def _load_normalizer(vecnormalize_path: Optional[str]) -> Optional[VecNormalize]:
    """Load a VecNormalize object purely to reuse its .normalize_obs()/
    .normalize_reward() math -- never call .reset()/.step() on the throwaway
    DummyVecEnv this wraps; the actual rollout below drives its own raw
    env directly so it can control the reset seed (see module docstring)."""
    if vecnormalize_path is None:
        return None
    throwaway = DummyVecEnv([lambda: gym.make(ENV_ID)])
    normalizer = VecNormalize.load(vecnormalize_path, throwaway)
    normalizer.training = False
    return normalizer


def run_one_episode(
    label: str,
    env_fn: Callable[[], gym.Env],
    model: PPO,
    normalizer: Optional[VecNormalize],
    seed: int,
    deterministic: bool = True,
) -> Trajectory:
    """Roll out exactly one episode against `env_fn()`, recording every
    step -- the single-episode, full-detail counterpart to
    evaluation/evaluate_policy.py::evaluate()'s many-episode aggregate."""
    env_config = EnvironmentConfig()
    traj = Trajectory(label=label)
    env = env_fn()

    obs, _info = env.reset(seed=seed)
    cumulative = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        model_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
        action, _ = model.predict(model_input, deterministic=deterministic)

        normalized_action = float(np.clip(action, -1.0, 1.0).reshape(-1)[0])
        torque_nm = normalized_action * env_config.max_motor_torque_nm

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += float(reward)

        traj.tank_angle.append(info["tank_angle"])
        traj.tank_angular_velocity.append(info["tank_angular_velocity"])
        traj.slosh_energy.append(info["slosh_energy"])
        traj.commanded_torque_nm.append(torque_nm)
        traj.reward.append(float(reward))
        traj.cumulative_reward.append(cumulative)

    env.close()
    logger.info(
        "%s: %d steps, return=%.4f, final_slosh_energy=%.6f",
        label, len(traj.reward), traj.total_return,
        traj.slosh_energy[-1] if traj.slosh_energy else float("nan"),
    )
    return traj


def _write_csv(path: str, trajectories: List[Trajectory]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source", "step", "tank_angle", "tank_angular_velocity",
            "slosh_energy", "commanded_torque_nm", "reward", "cumulative_reward",
        ])
        for traj in trajectories:
            for step in range(len(traj.reward)):
                writer.writerow([
                    traj.label, step, traj.tank_angle[step],
                    traj.tank_angular_velocity[step], traj.slosh_energy[step],
                    traj.commanded_torque_nm[step], traj.reward[step],
                    traj.cumulative_reward[step],
                ])
    logger.info("Wrote per-step CSV to %s", path)


def _write_plot(path: str, trajectories: List[Trajectory]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless -- this project has no display server assumption
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed -- skipping %s (CSV was still written). "
            "Install with: pip install matplotlib",
            path,
        )
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for traj in trajectories:
        steps = range(len(traj.reward))
        axes[0].plot(steps, traj.tank_angle, label=traj.label)
        axes[1].plot(steps, traj.slosh_energy, label=traj.label)
        axes[2].plot(steps, traj.cumulative_reward, label=traj.label)

    axes[0].set_ylabel("tank angle (rad)")
    axes[1].set_ylabel("slosh energy (J)")
    axes[2].set_ylabel("cumulative reward")
    axes[2].set_xlabel("control step")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Surrogate vs. real-physics trajectory comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote comparison plot to %s", path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Path to the saved PPO model (with or without .zip)")
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Path to the saved VecNormalize stats (defaults to '<model_path>_vecnormalize.pkl' if it exists)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic policy mean")
    parser.add_argument("--output-prefix", default="trajectory_comparison", help="Output files: <prefix>.csv and <prefix>.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        default_stats_path = args.model_path.rstrip(".zip") + "_vecnormalize.pkl"
        if os.path.exists(default_stats_path):
            vecnormalize_path = default_stats_path
            logger.info("Found VecNormalize stats at %s, using them.", vecnormalize_path)

    model = PPO.load(args.model_path)
    normalizer = _load_normalizer(vecnormalize_path)

    logger.info(
        "Running one surrogate episode and one REAL ChSystem episode, both "
        "seed=%d -- the real one will be much slower (see "
        "scripts/validate_real_dynamics.py's own docstring).",
        args.seed,
    )

    surrogate_traj = run_one_episode(
        "surrogate", lambda: gym.make(ENV_ID), model, normalizer, args.seed,
        deterministic=not args.stochastic,
    )
    real_traj = run_one_episode(
        "real_chrono", _build_real_dynamics_env, model, normalizer, args.seed,
        deterministic=not args.stochastic,
    )

    _write_csv(f"{args.output_prefix}.csv", [surrogate_traj, real_traj])
    _write_plot(f"{args.output_prefix}.png", [surrogate_traj, real_traj])

    gap = real_traj.total_return - surrogate_traj.total_return
    logger.info(
        "Return gap (real - surrogate): %.4f (surrogate=%.4f, real=%.4f)",
        gap, surrogate_traj.total_return, real_traj.total_return,
    )
