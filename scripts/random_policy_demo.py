"""
Phase 8 sanity check: run SloshingSuppressionEnv with a random policy.

Unlike main.py (Phase 6-7's Irrlicht demo), this needs no pychrono at
all -- it exercises the pure-Python environment/ layer only, so it runs
anywhere this project's pip requirements are installed.

Run:
    python -m scripts.random_policy_demo
"""

import logging

import numpy as np

import environment  # noqa: F401 -- import for its gym.register() side effect
import gymnasium as gym

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_NUM_EPISODES = 3


def main() -> None:
    env = gym.make("LiquidSloshingRL/SloshingSuppression-v0")
    rng = np.random.default_rng(0)

    for episode in range(_NUM_EPISODES):
        obs, info = env.reset(seed=episode)
        episode_return = 0.0
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1

        logger.info(
            "Episode %d: steps=%d return=%.4f final_slosh_energy=%.6f terminated=%s",
            episode, step_count, episode_return, info["slosh_energy"], terminated,
        )

    env.close()


if __name__ == "__main__":
    main()
