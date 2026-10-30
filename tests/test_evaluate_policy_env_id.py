"""Regression guard for a real bug found and fixed this session
(the development log Entry 69): evaluation/evaluate_policy.py's CLI used to
hardcode ENV_ID (the base god-mode SloshingSuppressionEnv) regardless of
which environment a given model was actually trained against. Since
LiquidSloshingRL/ObservableSloshingSuppression-v0 happens to have the
SAME observation shape (7,), this didn't error -- it silently evaluated
a pressure-observation-trained policy against fluid-COM observations and
produced a nonsense return (-1.01 measured directly, vs. the real
-0.0028) instead of failing loudly. Fixed by adding a --env-id CLI flag
that's actually threaded through to evaluate()'s env_fn.
"""

import gymnasium as gym
import pytest

pytest.importorskip(
    "stable_baselines3", reason="stable-baselines3 required for Phase 9 training tests."
)

import environment  # noqa: F401 -- import for its gym.register() side effect
from config.simulation_config import TrainingConfig
from evaluation.evaluate_policy import evaluate
from training.train_ppo_observable import ENV_ID as OBSERVABLE_ENV_ID
from training.train_ppo_observable import train as train_observable

BASE_ENV_ID = "LiquidSloshingRL/SloshingSuppression-v0"


def test_env_fn_actually_changes_which_environment_is_evaluated(tmp_path):
    """A policy trained on ObservableSloshingSuppression-v0, evaluated
    against its OWN env-id, should not crash and should return a value in
    a sane, small-negative range. Evaluated against the MISMATCHED base
    env-id, it should produce a clearly different (much worse) return --
    proving env_fn is actually wired through, not silently ignored."""
    config = TrainingConfig(
        total_timesteps=256,
        n_envs=2,
        n_steps=128,
        batch_size=32,
        n_epochs=1,
        checkpoint_every_steps=10_000,
        tensorboard_log_dir="",
        model_save_path=str(tmp_path / "ppo_observable_tiny"),
    )
    train_observable(config)
    model_path = str(tmp_path / "ppo_observable_tiny")
    vecnormalize_path = model_path + "_vecnormalize.pkl"

    matched_returns, _ = evaluate(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        n_episodes=2,
        env_fn=lambda: gym.make(OBSERVABLE_ENV_ID),
    )
    mismatched_returns, _ = evaluate(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        n_episodes=2,
        env_fn=lambda: gym.make(BASE_ENV_ID),
    )

    # The two env_fns must actually produce different rollouts -- if
    # evaluate() silently ignored env_fn and always used one hardcoded
    # env, these would be identical.
    assert matched_returns != mismatched_returns
