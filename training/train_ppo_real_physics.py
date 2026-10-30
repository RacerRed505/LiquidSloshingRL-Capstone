"""
Phase 11 -- Fine-tune the surrogate-trained PPO policy against the REAL
ChSystem-backed tank (environment/chrono_tank_dynamics.py::ChronoTankDynamics),
instead of training/train_ppo.py's fast analytic surrogate.

Context this exists for: scripts/validate_real_dynamics.py (Phase 10)
found a real, consistent gap between the two (mean return -0.0008
surrogate vs -0.0093 real physics, ~11.6x, final slosh energy ~0 in
both -- see CONTINUE_HERE.md's Phase 10 section). The policy still
damps the tank under real physics, it just costs more along the way --
so WARM-STARTING from the existing surrogate-trained model
(models/ppo_sloshing.zip) and fine-tuning briefly under real physics is
the natural next step, not training a fresh policy from scratch (which
would throw away 500k timesteps of already-useful surrogate training to
re-learn things real physics didn't actually get wrong -- damping in
the right direction, staying inside the angle limit, etc).

Why SubprocVecEnv, not train_ppo.py's DummyVecEnv:
DummyVecEnv runs every sub-env in the SAME process/thread, one after
another -- fine for the analytic surrogate (pure Python, no shared
mutable global state to worry about), but environment/
chrono_tank_dynamics.py's own docstring flags that PyChrono is not
free-threading-safe the way pure Python is. SubprocVecEnv instead spawns
one OS process per env, each importing pychrono and building its own
ChSystem independently -- true isolation, not just true parallelism.

Wall-clock expectations (read before picking --total-timesteps):
CONTINUE_HERE.md's Phase 10 section says main.py "ran close to
real-time," but that number comes from main.py's Irrlicht `vis.Run()`
loop, which paces itself to the render/vsync rate -- it is NOT a
measurement of how fast a headless `system.DoStepDynamics()` call
actually is on its own. This script (like scripts/validate_real_dynamics.py)
never opens a render window, so it may well run faster than that
figure implies -- but nobody has timed it yet. Don't guess: run with
--total-timesteps set very small first (the default below) and read the
actual `fps` PPO reports in its own training log before committing to a
longer run.

Run:
    python -m training.train_ppo_real_physics
    python -m training.train_ppo_real_physics --total-timesteps 5000 --n-envs 4
    python -m training.train_ppo_real_physics --from-scratch --total-timesteps 5000

Requires the real PyChrono bindings (conda-forge), same as main.py and
scripts/validate_real_dynamics.py:
    conda install -c conda-forge pychrono
This will NOT run in a sandbox/CI environment without them.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from typing import Callable, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from config.simulation_config import (
    EnvironmentConfig,
    PivotConfig,
    TankConfig,
    TrainingConfig,
    WorldConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _make_real_dynamics_env_fn() -> Callable[[], "gym.Env"]:  # noqa: F821 -- see below
    """Return a zero-arg callable that builds ONE fresh real-physics env.

    Deliberately does not import pychrono / chrono_tank_dynamics at this
    module's top level -- SubprocVecEnv cloudpickles this callable and
    calls it INSIDE each spawned worker process, which is where pychrono
    actually needs to be importable, not in the parent process that just
    constructs the list of these callables (mirrors how
    scripts/validate_real_dynamics.py already only needs pychrono at the
    point ChronoTankDynamics() is actually constructed).
    """

    def _init():
        import gymnasium as gym  # noqa: F401 -- imported for the return type only

        from environment.chrono_tank_dynamics import ChronoTankDynamics
        from environment.sloshing_env import SloshingSuppressionEnv

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

    return _init


def build_real_dynamics_vec_env(n_envs: int, normalize: bool, vecnormalize_seed_path: Optional[str]):
    """Build a SubprocVecEnv of `n_envs` real-physics envs, optionally
    wrapped in VecNormalize continuing from an existing model's saved
    running statistics (rather than starting fresh ones) -- so the
    warm-started policy keeps seeing observations on the same scale it
    was already trained on, instead of a jarring reset to raw stats."""
    env_fns: List[Callable] = [_make_real_dynamics_env_fn() for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    if not normalize:
        return vec_env

    if vecnormalize_seed_path is not None and os.path.exists(vecnormalize_seed_path):
        vec_env = VecNormalize.load(vecnormalize_seed_path, vec_env)
        vec_env.training = True  # keep updating running stats as fine-tuning proceeds
        vec_env.norm_reward = True
        logger.info(
            "Continuing VecNormalize running stats from %s (not starting fresh).",
            vecnormalize_seed_path,
        )
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        logger.warning(
            "No existing VecNormalize stats found at %r -- starting fresh running "
            "statistics. This is expected with --from-scratch, but if you meant to "
            "warm-start, double check --base-vecnormalize-path.",
            vecnormalize_seed_path,
        )
    return vec_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default="models/ppo_sloshing",
        help="Surrogate-trained model to warm-start from (ignored with --from-scratch)",
    )
    parser.add_argument(
        "--base-vecnormalize-path", default=None,
        help="Defaults to '<base-model>_vecnormalize.pkl' if it exists",
    )
    parser.add_argument("--from-scratch", action="store_true", help="Train a fresh PPO policy under real physics instead of warm-starting --base-model")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel real-physics envs (SubprocVecEnv) -- keep small, see this module's docstring")
    parser.add_argument("--total-timesteps", type=int, default=2000, help="Deliberately tiny default -- time a small run before raising this, see this module's docstring")
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--output", default="models/ppo_sloshing_realphysics", help="Where to save the fine-tuned model (does NOT overwrite --base-model)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def train(args: argparse.Namespace) -> PPO:
    base_vecnormalize_path = None
    if not args.from_scratch:
        # Only reuse the base model's running obs/reward stats when
        # actually warm-starting from it -- --from-scratch means fresh
        # stats too, otherwise the "FRESH policy" log line below would be
        # contradicted by silently-reused normalization statistics.
        base_vecnormalize_path = args.base_vecnormalize_path
        if base_vecnormalize_path is None:
            default_path = args.base_model.rstrip(".zip") + "_vecnormalize.pkl"
            if os.path.exists(default_path):
                base_vecnormalize_path = default_path

    vec_env = build_real_dynamics_vec_env(
        n_envs=args.n_envs, normalize=True, vecnormalize_seed_path=base_vecnormalize_path,
    )

    if args.from_scratch:
        # Same PPO hyperparameters train_ppo.py uses for the surrogate --
        # a from-scratch real-physics run should still be a fair
        # comparison against that baseline, not a different algorithm
        # config confounding the comparison.
        hp = TrainingConfig()
        model = PPO(
            "MlpPolicy", vec_env, n_steps=hp.n_steps, batch_size=min(hp.batch_size, args.n_envs * hp.n_steps),
            n_epochs=hp.n_epochs, learning_rate=hp.learning_rate, gamma=hp.gamma,
            gae_lambda=hp.gae_lambda, clip_range=hp.clip_range, ent_coef=hp.ent_coef,
            seed=args.seed, verbose=1,
        )
        logger.info("Training a FRESH policy under real physics (--from-scratch).")
    else:
        # NOTE: this must be PPO.load(path, env=...), NOT
        # PPO.load(path) followed by model.set_env(vec_env). set_env()
        # asserts the new env's num_envs matches the ALREADY-LOADED
        # model's n_envs (from the original Phase 9 training run, 8) --
        # it always fails here since --n-envs defaults to 4 specifically
        # to keep real-physics envs cheap (see this module's own
        # docstring), and is deterministic: it will fail on every run
        # with these inputs, not just sometimes. PPO.load(path, env=...)
        # is SB3's own documented way to load a model against an env with
        # a different num_envs (confirmed: it still restores
        # num_timesteps for `reset_num_timesteps=False` to continue
        # correctly below -- this was verified in the assistant's sandbox
        # after this bug surfaced on a real run, not assumed).
        model = PPO.load(args.base_model, env=vec_env)
        logger.info("Warm-starting from %s.zip under real physics.", args.base_model)

    save_dir = os.path.dirname(args.output) or "."
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_every_steps // args.n_envs, 1),
        save_path=save_dir,
        name_prefix=os.path.basename(args.output),
    )

    logger.info(
        "Starting real-physics PPO run: total_timesteps=%d n_envs=%d "
        "(SubprocVecEnv). Watch the fps PPO reports below -- that's your "
        "actual wall-clock budget for a longer run, not a guess.",
        args.total_timesteps, args.n_envs,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,  # continuing an existing policy's training, not starting a new counter
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
