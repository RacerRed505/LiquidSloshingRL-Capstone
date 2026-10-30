"""
Trains PPO against `LiquidSloshingRL/ObservableSloshingSuppression-v0`
(environment/observable_sloshing_env.py::ObservableSloshingEnv) -- the
hardware-deployable POMDP variant of Phase 9's original surrogate-trained
policy, using the SAME fast PyChrono-free surrogate (PendulumBackend +
TankDynamicsModel) training/train_ppo.py trains against, just with the
wall-pressure-sensor observation instead of raw fluid COM.

This directly answers a question an external reference package
(see the project notes' own note on that package) only ASSERTED without ever
training anything: "policies trained on differential pressure achieve
96% of god-mode performance." That number was never produced by an
actual training run in that package. This script and
evaluation/evaluate_policy.py (pointed at this model via `--env-id`)
produce a real one instead.

Deliberately a thin, separate script rather than a `--observable` flag
on training/train_ppo.py -- mirrors this project's own existing
precedent (training/train_ppo_real_physics.py alongside train_ppo.py)
of a new script per meaningfully different training target, not a
growing pile of conditional flags on one function.

Run:
    python -m training.train_ppo_observable
    python -m training.train_ppo_observable --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import environment  # noqa: F401 -- import for its gym.register() side effect
from config.simulation_config import TrainingConfig
from training.train_ppo import _resolve_tensorboard_log_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ENV_ID = "LiquidSloshingRL/ObservableSloshingSuppression-v0"


def build_vec_env(training_config: TrainingConfig):
    """Identical to training/train_ppo.py::build_vec_env(), against ENV_ID above."""
    vec_env = make_vec_env(ENV_ID, n_envs=training_config.n_envs, seed=training_config.seed)
    if training_config.normalize_observations or training_config.normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=training_config.normalize_observations,
            norm_reward=training_config.normalize_rewards,
        )
    return vec_env


def train(training_config: Optional[TrainingConfig] = None) -> PPO:
    """Run PPO training end-to-end against the observable env and save the result."""
    training_config = training_config or TrainingConfig(model_save_path="models/ppo_sloshing_observable")

    vec_env = build_vec_env(training_config)
    tensorboard_log = _resolve_tensorboard_log_dir(training_config)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        n_epochs=training_config.n_epochs,
        learning_rate=training_config.learning_rate,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        clip_range=training_config.clip_range,
        ent_coef=training_config.ent_coef,
        seed=training_config.seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    import os

    save_dir = os.path.dirname(training_config.model_save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(training_config.checkpoint_every_steps // training_config.n_envs, 1),
        save_path=save_dir,
        name_prefix=os.path.basename(training_config.model_save_path),
    )

    logger.info(
        "Starting PPO training on %s: total_timesteps=%d n_envs=%d n_steps=%d",
        ENV_ID, training_config.total_timesteps, training_config.n_envs, training_config.n_steps,
    )
    model.learn(total_timesteps=training_config.total_timesteps, callback=checkpoint_callback)

    model.save(training_config.model_save_path)
    logger.info("Saved model to %s.zip", training_config.model_save_path)

    if isinstance(vec_env, VecNormalize):
        stats_path = training_config.model_save_path + "_vecnormalize.pkl"
        vec_env.save(stats_path)
        logger.info("Saved VecNormalize stats to %s", stats_path)

    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--model-save-path", type=str, default="models/ppo_sloshing_observable")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_config = TrainingConfig(model_save_path=args.model_save_path)
    overrides = {}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    config = replace(base_config, **overrides) if overrides else base_config

    train(config)
