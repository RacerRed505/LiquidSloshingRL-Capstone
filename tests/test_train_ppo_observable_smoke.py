"""
Smoke test for training/train_ppo_observable.py.

Same guard/scope discipline as tests/test_train_ppo_smoke.py: guarded
with pytest.importorskip("stable_baselines3", ...), runs one genuinely
tiny PPO rollout/update purely to catch integration breakage
(observation-space mismatches, env registration typos) fast -- not a
convergence test. See the development log Entry 68's follow-up for the real
500k-timestep training run and its evaluation numbers.

Run:
    pytest tests/test_train_ppo_observable_smoke.py -v
"""

import pytest

pytest.importorskip(
    "stable_baselines3", reason="stable-baselines3 required for Phase 9 training tests."
)

from config.simulation_config import TrainingConfig
from training.train_ppo_observable import ENV_ID, train


def test_env_id_is_registered():
    import gymnasium as gym

    import environment  # noqa: F401 -- import for its gym.register() side effect

    env = gym.make(ENV_ID)
    assert env.observation_space.shape == (7,)
    env.close()


def test_ppo_trains_for_one_rollout_without_crashing(tmp_path):
    config = TrainingConfig(
        total_timesteps=128,
        n_envs=2,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_every_steps=1000,
        tensorboard_log_dir="",
        model_save_path=str(tmp_path / "ppo_observable_smoke"),
    )

    model = train(config)

    assert model is not None
    assert (tmp_path / "ppo_observable_smoke.zip").exists()
    assert (tmp_path / "ppo_observable_smoke_vecnormalize.pkl").exists()
