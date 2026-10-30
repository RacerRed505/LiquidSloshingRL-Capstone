"""
Phase 9 -- Evaluate a trained PPO policy.

Loads a saved stable-baselines3 PPO model (and its VecNormalize stats, if
training used normalization -- see TrainingConfig.normalize_observations/
normalize_rewards) and runs it deterministically against
`LiquidSloshingRL/SloshingSuppression-v0` for a fixed number of episodes,
reporting return and final slosh energy per episode.

Run:
    python -m evaluation.evaluate_policy models/ppo_sloshing
    python -m evaluation.evaluate_policy models/ppo_sloshing --n-episodes 20
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import environment  # noqa: F401 -- import for its gym.register() side effect

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENV_ID = "LiquidSloshingRL/SloshingSuppression-v0"


def evaluate(
    model_path: str,
    vecnormalize_path: Optional[str] = None,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
    env_fn: Optional[Callable[[], gym.Env]] = None,
) -> Tuple[List[float], List[float]]:
    """Run `n_episodes` deterministic episodes and return (returns, final_slosh_energies).

    If `vecnormalize_path` is given, observations are normalized with the
    SAME running statistics training produced -- evaluating a
    VecNormalize-trained policy against raw, unnormalized observations
    would silently feed it out-of-distribution inputs.

    `env_fn`, added Phase 10, lets a caller substitute a differently
    constructed env (e.g. scripts/validate_real_dynamics.py, which
    injects environment.chrono_tank_dynamics.ChronoTankDynamics instead
    of the default analytic surrogate) without duplicating this
    function's episode-rollout/VecNormalize-handling logic. Defaults to
    `lambda: gym.make(ENV_ID)`, i.e. unchanged behavior for existing
    callers.
    """
    if env_fn is None:
        env_fn = lambda: gym.make(ENV_ID)  # noqa: E731

    model = PPO.load(model_path)

    if vecnormalize_path is not None:
        vec_env = DummyVecEnv([env_fn])
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False  # freeze running statistics
        vec_env.norm_reward = False  # report raw reward, not the normalized training signal

        episode_returns: List[float] = []
        episode_final_slosh: List[float] = []
        for episode in range(n_episodes):
            obs = vec_env.reset()
            terminated_or_truncated = False
            episode_return = 0.0
            info: dict = {}
            while not terminated_or_truncated:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = vec_env.step(action)
                episode_return += float(reward[0])
                terminated_or_truncated = bool(done[0])
                info = infos[0]
            episode_returns.append(episode_return)
            episode_final_slosh.append(info["slosh_energy"])
            logger.info(
                "Episode %d: return=%.4f final_slosh_energy=%.6f",
                episode, episode_return, info["slosh_energy"],
            )
    else:
        env = env_fn()
        episode_returns = []
        episode_final_slosh = []
        for episode in range(n_episodes):
            obs, info = env.reset(seed=seed + episode)
            terminated = truncated = False
            episode_return = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
            episode_returns.append(episode_return)
            episode_final_slosh.append(info["slosh_energy"])
            logger.info(
                "Episode %d: return=%.4f final_slosh_energy=%.6f",
                episode, episode_return, info["slosh_energy"],
            )
        env.close()

    logger.info(
        "Mean return: %.4f +/- %.4f | Mean final slosh energy: %.6f +/- %.6f",
        np.mean(episode_returns), np.std(episode_returns),
        np.mean(episode_final_slosh), np.std(episode_final_slosh),
    )
    return episode_returns, episode_final_slosh


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Path to the saved PPO model (with or without .zip)")
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Path to the saved VecNormalize stats (defaults to '<model_path>_vecnormalize.pkl' if it exists)",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic policy mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-id",
        default=ENV_ID,
        help=(
            "Gymnasium env ID to evaluate against (default: the base god-mode "
            "SloshingSuppressionEnv). A policy trained against a DIFFERENT "
            "observation space (e.g. models/ppo_sloshing_observable, trained "
            "against LiquidSloshingRL/ObservableSloshingSuppression-v0) MUST "
            "pass the matching --env-id here -- both envs currently happen to "
            "have the same observation shape (7,), so a mismatch silently "
            "produces a nonsense return instead of an error. Confirmed "
            "directly: running ppo_sloshing_observable against the default "
            "env-id here gives return=-1.01, vs. its real -0.0028 against its "
            "own env-id -- see the development log Entry 69."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        default_stats_path = args.model_path.rstrip(".zip") + "_vecnormalize.pkl"
        if os.path.exists(default_stats_path):
            vecnormalize_path = default_stats_path
            logger.info("Found VecNormalize stats at %s, using them.", vecnormalize_path)

    evaluate(
        model_path=args.model_path,
        vecnormalize_path=vecnormalize_path,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        env_fn=lambda: gym.make(args.env_id),
    )
