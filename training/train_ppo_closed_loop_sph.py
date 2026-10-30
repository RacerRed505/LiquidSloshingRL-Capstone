"""
Phase 15 -- Fine-tune the surrogate/real-physics-trained PPO policy against
the REAL, two-way-coupled Chrono::FSI closed loop
(environment/closed_loop_sph_env.py::ClosedLoopSPHEnv), instead of just
running an already-trained policy against it open-loop
(scripts/run_closed_loop_sph_policy.py).

Context this exists for (the development log Entries 55-56): the closed loop's
real failure mode is a sudden, genuine control breakdown after an extended
period of good behavior -- confirmed NOT to be a pure numerics bug (higher
artificial_viscosity prevented the SPH solver crash but not the underlying
loss of control, Entry 56). Torque-limit/SPH-parameter tuning alone found a
real ~5x improvement (from ~40-50 steps to ~200-300 steps survival at
torque_limit=7.0) but plateaued well short of a reliable full 500-step run.
This script exists to let the policy itself adapt to the real coupled
dynamics directly, the same way Phase 11's train_ppo_real_physics.py closed
the gap between the surrogate and the real (non-FSI) ChSystem tank.

Why n_envs=1 (DummyVecEnv), NOT train_ppo_real_physics.py's SubprocVecEnv
pattern: Chrono::FSI needs a real CUDA GPU (Entries 50-54) -- running
multiple parallel FSI scenes on ONE consumer GPU (8GB VRAM here) risks VRAM
contention/OOM in a way pure-CPU rigid-body ChronoTankDynamics never did.
Starting with n_envs=1 is the safe default for a first real attempt; if a
real, measured throughput number later justifies it, n_envs could be raised
and this reconsidered -- not assumed safe ahead of time.

Wall-clock reality, read before picking --total-timesteps (do not guess):
ClosedLoopSPHTank steps at roughly 0.7s/control-step (Entries 54-56's own
measurements, ~35x slower than real-time) -- two to three orders of
magnitude slower per step than ChronoTankDynamics' pure-CPU rigid-body
stepping that made train_ppo_real_physics.py's SubprocVecEnv/n_envs=4
approach viable. At TrainingConfig's own default `n_steps=1024`, ONE
rollout alone would take ~1024*0.7s =~ 12 MINUTES before any policy update
happens at all -- `--n-steps` below defaults to something far smaller
(128) specifically so a full rollout+update cycle finishes in a few
minutes, not to match train_ppo.py's own tuned value. Run with a small
--total-timesteps FIRST and read the real elapsed wall-clock this script
reports before committing to a longer budget.

Run:
    python -m training.train_ppo_closed_loop_sph --total-timesteps 256
    python -m training.train_ppo_closed_loop_sph --base-model models/ppo_sloshing_realphysics --total-timesteps 1024

Requires the real local Chrono::FSI build (the development log Entries 50-54) --
see the project notes' "Second machine note" for the exact PATH/PYTHONPATH needed
before this will import pychrono.fsi at all.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig, TrainingConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _make_closed_loop_env_fn(reset_angular_velocity_rad_s: float, artificial_viscosity: float, torque_limit: float, boundary_method: str = "adami") -> Callable[[], "gym.Env"]:  # noqa: F821
    """Return a zero-arg callable that builds ONE fresh ClosedLoopSPHEnv.

    Deliberately does not import pychrono.fsi / ClosedLoopSPHEnv at this
    module's top level -- same reasoning as
    training/train_ppo_real_physics.py::_make_real_dynamics_env_fn, so
    this module itself stays importable (e.g. for --help) even where
    pychrono.fsi isn't -- though with n_envs=1/DummyVecEnv the import
    actually happens in THIS process either way, unlike SubprocVecEnv's
    worker-process isolation.
    """

    def _init():
        from environment.closed_loop_sph_env import ClosedLoopSPHEnv

        env_config = EnvironmentConfig(max_motor_torque_nm=torque_limit)
        return ClosedLoopSPHEnv(
            tank_config=TankConfig(fixed=False),
            fluid_config=FluidConfig(),
            env_config=env_config,
            reset_angular_velocity_rad_s=reset_angular_velocity_rad_s,
            artificial_viscosity=artificial_viscosity,
            boundary_method=boundary_method,
        )

    return _init


def build_closed_loop_vec_env(
    reset_angular_velocity_rad_s: float, artificial_viscosity: float, torque_limit: float,
    normalize: bool, vecnormalize_seed_path: Optional[str],
    boundary_method: str = "adami",
):
    """Build a single-env DummyVecEnv (see this module's own docstring for
    why not SubprocVecEnv/n_envs>1), optionally wrapped in VecNormalize
    continuing from an existing model's saved running statistics."""
    env_fns: List[Callable] = [
        _make_closed_loop_env_fn(reset_angular_velocity_rad_s, artificial_viscosity, torque_limit, boundary_method)
    ]
    vec_env = DummyVecEnv(env_fns)

    if not normalize:
        return vec_env

    if vecnormalize_seed_path is not None and os.path.exists(vecnormalize_seed_path):
        vec_env = VecNormalize.load(vecnormalize_seed_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        logger.info(
            "Continuing VecNormalize running stats from %s (not starting fresh).",
            vecnormalize_seed_path,
        )
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        logger.warning(
            "No existing VecNormalize stats found at %r -- starting fresh running "
            "statistics.", vecnormalize_seed_path,
        )
    return vec_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="models/ppo_sloshing", help="Model to warm-start from (ignored with --from-scratch)")
    parser.add_argument("--base-vecnormalize-path", default=None, help="Defaults to '<base-model>_vecnormalize.pkl' if it exists")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--torque-limit", type=float, default=7.0, help="max_motor_torque_nm for THIS fine-tuning run -- see the development log Entry 56 for why 7.0, not the original 2.0, is the informed starting point.")
    parser.add_argument("--reset-velocity", type=float, default=0.05)
    parser.add_argument("--artificial-viscosity", type=float, default=0.03)
    parser.add_argument("--boundary-method", choices=["adami", "holmes"], default="adami", help="Chrono::FSI boundary-marker method. 'adami' matches every existing checkpoint's training dynamics -- unchanged unless explicitly overridden. 'holmes' trains under the alternative boundary method (the development log Entry 81 found it reduces wall-penetration ~8.5x but hurts survival for an ADAMI-trained policy evaluated under it; this flag is what makes a FAIR from-scratch HOLMES training run possible, added Entry 119).")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout length -- deliberately far smaller than TrainingConfig's own 1024, see this module's docstring on wall-clock cost.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=256, help="Deliberately tiny default -- time a small run before raising this.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=256)
    parser.add_argument("--output", default="models/ppo_sloshing_closed_loop_sph")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train(args: argparse.Namespace) -> PPO:
    base_vecnormalize_path = None
    if not args.from_scratch:
        base_vecnormalize_path = args.base_vecnormalize_path
        if base_vecnormalize_path is None:
            default_path = args.base_model.rstrip(".zip") + "_vecnormalize.pkl"
            if os.path.exists(default_path):
                base_vecnormalize_path = default_path

    vec_env = build_closed_loop_vec_env(
        reset_angular_velocity_rad_s=args.reset_velocity,
        artificial_viscosity=args.artificial_viscosity,
        torque_limit=args.torque_limit,
        normalize=True,
        vecnormalize_seed_path=base_vecnormalize_path,
        boundary_method=args.boundary_method,
    )

    if args.from_scratch:
        hp = TrainingConfig()
        model = PPO(
            "MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=min(args.batch_size, args.n_steps),
            n_epochs=hp.n_epochs, learning_rate=hp.learning_rate, gamma=hp.gamma,
            gae_lambda=hp.gae_lambda, clip_range=hp.clip_range, ent_coef=hp.ent_coef,
            seed=args.seed, verbose=1,
        )
        logger.info("Training a FRESH policy against the real closed-loop SPH tank (--from-scratch).")
    else:
        # PPO.load(path, env=...), not load()+set_env() -- same real bug
        # already found and fixed for the exact same reason in
        # train_ppo_real_physics.py (Entry 11): set_env() asserts
        # num_envs matches the ALREADY-LOADED model's n_envs (8, from the
        # original Phase 9 training), which n_envs=1 here always violates.
        model = PPO.load(args.base_model, env=vec_env)
        model.n_steps = args.n_steps
        model.rollout_buffer.buffer_size = args.n_steps
        model.batch_size = min(args.batch_size, args.n_steps)
        logger.info("Warm-starting from %s.zip against the real closed-loop SPH tank.", args.base_model)

    save_dir = os.path.dirname(args.output) or "."
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every_steps, 1),
        save_path=save_dir,
        name_prefix=os.path.basename(args.output),
        # the development log Entry 58: a real native-process crash (unrecoverable,
        # not catchable in Python) killed a 16k-timestep run mid-rollout --
        # only the periodic model .zip was saved, not VecNormalize's running
        # stats, since those were previously only written once at the very
        # end of a successful model.learn(). Saving them alongside every
        # checkpoint means a crash never loses more than checkpoint_every_steps
        # of VecNormalize progress either.
        save_vecnormalize=True,
    )

    logger.info(
        "Starting closed-loop SPH PPO fine-tune: total_timesteps=%d n_steps=%d "
        "torque_limit=%.1f (n_envs=1, DummyVecEnv). Watch the fps PPO reports "
        "below -- that's your actual wall-clock budget, not a guess.",
        args.total_timesteps, args.n_steps, args.torque_limit,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
    )

    model.save(args.output)
    logger.info("Saved fine-tuned model to %s.zip", args.output)

    if isinstance(vec_env, VecNormalize):
        stats_path = args.output + "_vecnormalize.pkl"
        vec_env.save(stats_path)
        logger.info("Saved VecNormalize stats to %s", stats_path)

    return model


if __name__ == "__main__":
    train(_parse_args())
