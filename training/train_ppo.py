"""
Phase 9 -- PPO training entry point.

Trains stable-baselines3's PPO against
`LiquidSloshingRL/SloshingSuppression-v0` (environment/sloshing_env.py),
using `n_envs` parallel copies (SB3's VecEnv), each running the Phase 8
pure-Python PendulumBackend + TankDynamicsModel surrogate -- no pychrono
anywhere in this training loop, which is exactly why parallelizing it
this way is cheap (see TrainingConfig.n_envs's own docstring).

Run:
    python -m training.train_ppo

Requires stable-baselines3 (pip installable, unlike pychrono):
    pip install stable-baselines3
TensorBoard logging is optional -- see TrainingConfig.tensorboard_log_dir.
"""

from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import environment  # noqa: F401 -- import for its gym.register() side effect
from config.simulation_config import TrainingConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ENV_ID = "LiquidSloshingRL/SloshingSuppression-v0"


def _resolve_tensorboard_log_dir(training_config: TrainingConfig) -> Optional[str]:
    """Only pass a tensorboard_log path through if the optional package is present.

    stable-baselines3 accepts a tensorboard_log path at PPO construction
    time with no complaint, then raises ImportError inside model.learn()
    itself if the `tensorboard` package isn't actually installed -- so
    this has to be checked up front rather than left to fail mid-training.
    """
    if not training_config.tensorboard_log_dir:
        return None
    if find_spec("tensorboard") is None:
        logger.warning(
            "TrainingConfig.tensorboard_log_dir=%r was set but the optional "
            "`tensorboard` package is not installed -- proceeding WITHOUT "
            "TensorBoard logging rather than crashing mid-training. "
            "Run `pip install tensorboard` to enable it.",
            training_config.tensorboard_log_dir,
        )
        return None
    os.makedirs(training_config.tensorboard_log_dir, exist_ok=True)
    return training_config.tensorboard_log_dir


def build_vec_env(training_config: TrainingConfig):
    """Build the (optionally normalized) parallel training environment."""
    vec_env = make_vec_env(ENV_ID, n_envs=training_config.n_envs, seed=training_config.seed)
    if training_config.normalize_observations or training_config.normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=training_config.normalize_observations,
            norm_reward=training_config.normalize_rewards,
        )
    return vec_env


def train(training_config: Optional[TrainingConfig] = None) -> PPO:
    """Run PPO training end-to-end and save the final model (+ VecNormalize stats)."""
    training_config = training_config or TrainingConfig()

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

    save_dir = os.path.dirname(training_config.model_save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(training_config.checkpoint_every_steps // training_config.n_envs, 1),
        save_path=save_dir,
        name_prefix=os.path.basename(training_config.model_save_path),
    )

    logger.info(
        "Starting PPO training: total_timesteps=%d n_envs=%d n_steps=%d",
        training_config.total_timesteps, training_config.n_envs, training_config.n_steps,
    )
    model.learn(total_timesteps=training_config.total_timesteps, callback=checkpoint_callback)

    model.save(training_config.model_save_path)
    logger.info("Saved model to %s.zip", training_config.model_save_path)

    if isinstance(vec_env, VecNormalize):
        stats_path = training_config.model_save_path + "_vecnormalize.pkl"
        vec_env.save(stats_path)
        logger.info("Saved VecNormalize stats to %s", stats_path)

    return model


if __name__ == "__main__":
    train()
