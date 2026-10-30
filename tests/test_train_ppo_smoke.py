"""
Smoke test for training/train_ppo.py (Phase 9).

Guarded with pytest.importorskip("stable_baselines3", ...) -- unlike
gymnasium, this is a heavier optional dependency, same defensive pattern
already used for pychrono elsewhere in this test suite. Where it IS
installed (pip install stable-baselines3, no conda-forge needed), this
runs a genuinely tiny training loop (one PPO rollout/update) purely to
catch integration breakage (bad shapes, config wiring mistakes) fast --
it is not a convergence or reward-quality test; see
evaluation/evaluate_policy.py for actually judging a trained policy.

Run:
    pytest tests/test_train_ppo_smoke.py -v
"""

import pytest

pytest.importorskip(
    "stable_baselines3", reason="stable-baselines3 required for Phase 9 training tests."
)

from config.simulation_config import TrainingConfig
from training.train_ppo import train


def test_ppo_trains_for_one_rollout_without_crashing(tmp_path):
    # n_envs * n_steps = 2 * 64 = 128 timesteps -- exactly one PPO
    # rollout/update, and total_timesteps=128 stops training right after
    # it. tensorboard_log_dir="" avoids requiring the optional
    # `tensorboard` package just to run this smoke test.
    config = TrainingConfig(
        total_timesteps=128,
        n_envs=2,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_every_steps=1000,
        tensorboard_log_dir="",
        model_save_path=str(tmp_path / "ppo_smoke"),
    )

    model = train(config)

    assert model is not None
    assert (tmp_path / "ppo_smoke.zip").exists()
    assert (tmp_path / "ppo_smoke_vecnormalize.pkl").exists()


def test_rejects_batch_size_larger_than_rollout_buffer():
    with pytest.raises(ValueError):
        TrainingConfig(n_envs=2, n_steps=64, batch_size=256)
