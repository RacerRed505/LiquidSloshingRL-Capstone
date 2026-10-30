"""
Phase 10 -- Validate a surrogate-trained PPO policy against the REAL
ChSystem-backed tank (environment/chrono_tank_dynamics.py), instead of
environment/tank_dynamics.py's analytic surrogate it was trained against.

This is deliberately the FIRST thing to run in Phase 10, before deciding
whether a slow real-physics retrain is even needed -- see
environment/chrono_tank_dynamics.py's own docstring for why. It reuses
evaluation/evaluate_policy.py::evaluate()'s episode-rollout logic via its
`env_fn` argument (added this session), so the only new code here is
constructing the right env.

Requires the REAL PyChrono bindings (conda-forge), same as main.py:
    conda install -c conda-forge pychrono
This will NOT run in a sandbox/CI environment without them -- there's no
pytest.importorskip guard here because this is a script, not a test, but
it will raise ImportError immediately on `import pychrono.core` if
PyChrono isn't installed.

Expect this to be dramatically slower per episode than
evaluate_policy.py's surrogate-based run -- each control step now costs a
real ChSystem.DoStepDynamics() call (physics_dt=1ms, 20 substeps per
20ms control step, matching EnvironmentConfig.substeps_per_control_step),
not a closed-form analytic update. main.py ran close to real-time for a
single instance; budget roughly that order of magnitude here too (an
n_episodes default of 5, not evaluate_policy.py's 10, for exactly this
reason -- raise it once you've seen how long one episode actually takes
on your machine).

Run:
    python -m scripts.validate_real_dynamics models/ppo_sloshing
    python -m scripts.validate_real_dynamics models/ppo_sloshing --n-episodes 3
"""

from __future__ import annotations

import argparse
import logging
import os

from config.simulation_config import EnvironmentConfig, PivotConfig, TankConfig, WorldConfig
from environment.chrono_tank_dynamics import ChronoTankDynamics
from environment.sloshing_env import SloshingSuppressionEnv
from evaluation.evaluate_policy import evaluate

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _build_real_dynamics_env() -> SloshingSuppressionEnv:
    """One env instance, real ChSystem tank + the same analytic PendulumBackend.

    Only the tank side swaps to real physics this phase -- the fluid
    stays the Phase 7 Housner surrogate (PendulumBackend), consistent
    with environment/chrono_tank_dynamics.py's own scope note that
    Phase 13-14 (SPHBackend) is a separate, later swap.
    """
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Path to the saved PPO model (with or without .zip)")
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Path to the saved VecNormalize stats (defaults to '<model_path>_vecnormalize.pkl' if it exists)",
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic policy mean")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        default_stats_path = args.model_path.rstrip(".zip") + "_vecnormalize.pkl"
        if os.path.exists(default_stats_path):
            vecnormalize_path = default_stats_path
            logger.info("Found VecNormalize stats at %s, using them.", vecnormalize_path)

    logger.info(
        "Running against the REAL ChSystem-backed tank -- this will be much "
        "slower per episode than evaluate_policy.py. See this script's own "
        "docstring for the expected order of magnitude."
    )

    evaluate(
        model_path=args.model_path,
        vecnormalize_path=vecnormalize_path,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
        env_fn=_build_real_dynamics_env,
    )
