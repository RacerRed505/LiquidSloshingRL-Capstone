"""
Phase 15 -- Fine-tune the surrogate-trained, hardware-deployable-observation
PPO policy (models/ppo_sloshing_observable.zip, trained against
environment/observable_sloshing_env.py::ObservableSloshingEnv) against the
REAL, two-way-coupled Chrono::FSI closed loop with the SAME pressure-based
observation (environment/observable_closed_loop_sph_env.py::
ObservableClosedLoopSPHEnv), instead of the raw-fluid-COM "god-mode"
observation training/train_ppo_closed_loop_sph.py fine-tunes.

This is a near-exact mirror of training/train_ppo_closed_loop_sph.py --
same rationale for n_envs=1/DummyVecEnv, same wall-clock-per-step reality
(read that module's own docstring, it applies unchanged here: ClosedLoopSPHTank
itself doesn't care what observation wrapper sits on top of it), same
CheckpointCallback(save_vecnormalize=True) crash-resilience convention. The
only real differences: the env class constructed (ObservableClosedLoopSPHEnv,
not ClosedLoopSPHEnv) and the default --base-model (the surrogate-trained
OBSERVABLE policy, ppo_sloshing_observable, not the god-mode ppo_sloshing --
warm-starting from a policy already trained on this same observation LAYOUT,
mirroring exactly how train_ppo_closed_loop_sph.py itself warm-starts from
the surrogate god-mode policy rather than training from scratch).

Run (small first, per this project's own "don't guess wall-clock" rule --
see train_ppo_closed_loop_sph.py's own docstring):
    python -m training.train_ppo_observable_closed_loop_sph --total-timesteps 256

Or via the crash-resilient supervisor (recommended for any real budget --
this backend has known, real, unrecoverable native crashes, see
scripts/supervised_closed_loop_finetune.py's own docstring):
    python -m scripts.supervised_closed_loop_finetune \
        --training-module training.train_ppo_observable_closed_loop_sph \
        --base-model models/ppo_sloshing_observable \
        --additional-timesteps 4096 --checkpoint-dir models \
        --checkpoint-prefix ppo_sloshing_observable_closed_loop_sph_campaign \
        --n-steps 256 --torque-limit 7.0 --artificial-viscosity 0.1 \
        --checkpoint-every-steps 256 --max-restarts 15
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


def _make_observable_closed_loop_env_fn(reset_angular_velocity_rad_s: float, artificial_viscosity: float, torque_limit: float, boundary_method: str = "adami") -> Callable[[], "gym.Env"]:  # noqa: F821
    """Same lazy-import pattern as train_ppo_closed_loop_sph.py's own
    _make_closed_loop_env_fn -- keeps this module importable (--help) even
    where pychrono.fsi isn't, though with n_envs=1/DummyVecEnv the import
    happens in this process either way."""

    def _init():
        from environment.observable_closed_loop_sph_env import ObservableClosedLoopSPHEnv

        env_config = EnvironmentConfig(max_motor_torque_nm=torque_limit)
        return ObservableClosedLoopSPHEnv(
            tank_config=TankConfig(fixed=False),
            fluid_config=FluidConfig(),
            env_config=env_config,
            reset_angular_velocity_rad_s=reset_angular_velocity_rad_s,
            artificial_viscosity=artificial_viscosity,
            boundary_method=boundary_method,
        )

    return _init


def build_observable_closed_loop_vec_env(
    reset_angular_velocity_rad_s: float, artificial_viscosity: float, torque_limit: float,
    normalize: bool, vecnormalize_seed_path: Optional[str],
    boundary_method: str = "adami",
):
    env_fns: List[Callable] = [
        _make_observable_closed_loop_env_fn(reset_angular_velocity_rad_s, artificial_viscosity, torque_limit, boundary_method)
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
    parser.add_argument("--base-model", default="models/ppo_sloshing_observable", help="Model to warm-start from (ignored with --from-scratch). Defaults to the SURROGATE-trained observable-observation policy, not the god-mode one -- see this module's own docstring.")
    parser.add_argument("--base-vecnormalize-path", default=None, help="Defaults to '<base-model>_vecnormalize.pkl' if it exists")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--torque-limit", type=float, default=7.0, help="max_motor_torque_nm for THIS fine-tuning run -- see the development log Entry 56 for why 7.0, not the original 2.0, is the informed starting point.")
    parser.add_argument("--reset-velocity", type=float, default=0.05)
    parser.add_argument("--artificial-viscosity", type=float, default=0.03)
    parser.add_argument("--boundary-method", choices=["adami", "holmes"], default="adami", help="Chrono::FSI boundary-marker method -- see train_ppo_closed_loop_sph.py's own --boundary-method help for the full story (the development log Entry 81/119). Added here for parity so scripts/supervised_closed_loop_finetune.py's unconditional --boundary-method forwarding doesn't break this training module too.")
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout length -- see train_ppo_closed_loop_sph.py's own docstring on wall-clock cost.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=256, help="Deliberately tiny default -- time a small run before raising this.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=256)
    parser.add_argument("--output", default="models/ppo_sloshing_observable_closed_loop_sph")
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

    vec_env = build_observable_closed_loop_vec_env(
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
        logger.info("Training a FRESH policy against the real observable closed-loop SPH tank (--from-scratch).")
    else:
        # PPO.load(path, env=...), not load()+set_env() -- same reasoning as
        # train_ppo_closed_loop_sph.py/train_ppo_real_physics.py (Entry 11).
        model = PPO.load(args.base_model, env=vec_env)
        model.n_steps = args.n_steps
        model.rollout_buffer.buffer_size = args.n_steps
        model.batch_size = min(args.batch_size, args.n_steps)
        logger.info("Warm-starting from %s.zip against the real observable closed-loop SPH tank.", args.base_model)

    save_dir = os.path.dirname(args.output) or "."
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every_steps, 1),
        save_path=save_dir,
        name_prefix=os.path.basename(args.output),
        save_vecnormalize=True,
    )

    logger.info(
        "Starting observable closed-loop SPH PPO fine-tune: total_timesteps=%d n_steps=%d "
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
