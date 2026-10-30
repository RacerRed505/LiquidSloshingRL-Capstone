"""
Phase 15 -- checkpoint-quality sweep for the observable (hardware-deployable,
pressure-based) closed-loop SPH campaign
(training.train_ppo_observable_closed_loop_sph /
scripts.supervised_closed_loop_finetune), mirroring the god-mode campaign's
own Entry 79 sweep methodology.

That earlier sweep found a real, non-obvious result for the god-mode
campaign: the campaign's own FINAL checkpoint (524288) was clearly worse
than an earlier one (514560), which turned out to be the best usable
policy of all 25 checkpoints tested. This script asks the same question
for the observable campaign, which just finished its own 16,384-step
run (base 507904 -> final 524288) crash-free but with NO confirmed
control-quality evaluation yet -- do not assume the final checkpoint is
the best one here either.

Reuses the REAL ObservableClosedLoopSPHEnv (via the exact same
DummyVecEnv+VecNormalize construction training.train_ppo_observable_closed_loop_sph.py
itself uses) rather than reimplementing the pressure observation, so this
can't drift from what each checkpoint was actually trained against.

Needs the real local Chrono::FSI build/CUDA GPU -- cannot run on this
project's original dev machine (no CUDA, see the project notes' "Environment
setup"/"Second machine note"). Run on Colab or the second RTX-GPU machine,
after the usual ctypes preload sequence.

Run (from repo root, one call sweeps every checkpoint matching the glob,
sorted by step count):
    python -m scripts.sweep_observable_closed_loop_checkpoints \
        --checkpoint-glob "models/ppo_sloshing_observable_closed_loop_sph_campaign_*_steps.zip" \
        --n-steps 250 --torque-limit 7.0 --artificial-viscosity 0.1 --reset-velocity 0.05

To only sweep this session's NEW checkpoints (512000 onward), not the
already-evaluated earlier ones:
    python -m scripts.sweep_observable_closed_loop_checkpoints --min-step 512000
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import Dict, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig

# WARNING (not INFO) so a sweep across many checkpoints isn't drowned in the
# per-10-steps INFO lines ClosedLoopSPHEnv.step() logs -- the mid-step/
# between-step safety-trip WARNINGs (the actually interesting events) still
# print.
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

_STEP_SUFFIX_RE = re.compile(r"_(\d+)_steps\.zip$")


def _make_env_fn(reset_angular_velocity_rad_s: float, artificial_viscosity: float, torque_limit: float):
    """Lazy-import so this module stays importable (--help) without pychrono.fsi."""

    def _init():
        from environment.observable_closed_loop_sph_env import ObservableClosedLoopSPHEnv

        env_config = EnvironmentConfig(max_motor_torque_nm=torque_limit)
        return ObservableClosedLoopSPHEnv(
            tank_config=TankConfig(fixed=False),
            fluid_config=FluidConfig(),
            env_config=env_config,
            reset_angular_velocity_rad_s=reset_angular_velocity_rad_s,
            artificial_viscosity=artificial_viscosity,
        )

    return _init


def _checkpoint_step_count(path: str) -> int:
    match = _STEP_SUFFIX_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def _vecnormalize_path_for(model_path: str, steps: int) -> str:
    # "..._campaign_508160_steps.zip" -> "..._campaign_vecnormalize_508160_steps.pkl"
    # -- matches CheckpointCallback(save_vecnormalize=True)'s own naming.
    prefix = _STEP_SUFFIX_RE.sub("", model_path)
    return f"{prefix}_vecnormalize_{steps}_steps.pkl"


def evaluate_checkpoint(
    model_path: str,
    vecnormalize_path: str,
    n_steps: int,
    torque_limit: float,
    artificial_viscosity: float,
    reset_velocity: float,
    seed: int,
) -> Dict[str, float]:
    vec_env = DummyVecEnv(
        [_make_env_fn(reset_velocity, artificial_viscosity, torque_limit)]
    )
    if os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    else:
        logger.warning(
            "No VecNormalize stats found at %s -- evaluating with FRESH stats "
            "(results not comparable to training-time normalization).",
            vecnormalize_path,
        )
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)
    obs = vec_env.reset()
    vec_env.seed(seed)

    survived_steps = 0
    final_angular_velocity = 0.0
    final_slosh_energy = 0.0
    max_slosh_energy = 0.0
    terminated_early = False

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, infos = vec_env.step(action)
        info = infos[0]
        survived_steps = step + 1
        final_angular_velocity = info["tank_angular_velocity"]
        final_slosh_energy = info["slosh_energy"]
        max_slosh_energy = max(max_slosh_energy, final_slosh_energy)
        if done[0]:
            terminated_early = survived_steps < n_steps
            break

    vec_env.close()
    return {
        "survived_steps": survived_steps,
        "final_angular_velocity": final_angular_velocity,
        "final_slosh_energy": final_slosh_energy,
        "max_slosh_energy": max_slosh_energy,
        "terminated_early": terminated_early,
    }


def evaluate_passive(
    n_steps: int,
    torque_limit: float,
    artificial_viscosity: float,
    reset_velocity: float,
    seed: int,
) -> Dict[str, float]:
    """Zero-torque baseline, same env/disturbance settings as evaluate_checkpoint,
    no policy/VecNormalize involved at all. Exists to answer a real question the
    checkpoint sweep alone can't: if every checkpoint trips the safety limit at
    roughly the same step regardless of how much fine-tuning it received, is
    that a policy-quality finding, or is this specific disturbance
    (reset_velocity/torque_limit combination) simply not survivable by ANY
    policy within this control-authority budget? If the passive baseline ALSO
    trips around the same step, the checkpoint sweep's flat result is
    explained by the physics/settings, not by the campaign underperforming."""
    env = _make_env_fn(reset_velocity, artificial_viscosity, torque_limit)()
    import numpy as np

    obs, _info = env.reset(seed=seed)
    survived_steps = 0
    final_angular_velocity = 0.0
    final_slosh_energy = 0.0
    max_slosh_energy = 0.0
    terminated_early = False

    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    for step in range(n_steps):
        _obs, _reward, terminated, truncated, info = env.step(zero_action)
        survived_steps = step + 1
        final_angular_velocity = info["tank_angular_velocity"]
        final_slosh_energy = info["slosh_energy"]
        max_slosh_energy = max(max_slosh_energy, final_slosh_energy)
        if terminated or truncated:
            terminated_early = survived_steps < n_steps
            break

    env.close()
    return {
        "survived_steps": survived_steps,
        "final_angular_velocity": final_angular_velocity,
        "final_slosh_energy": final_slosh_energy,
        "max_slosh_energy": max_slosh_energy,
        "terminated_early": terminated_early,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-glob",
        default="models/ppo_sloshing_observable_closed_loop_sph_campaign_*_steps.zip",
    )
    parser.add_argument("--n-steps", type=int, default=250, help="Same window Entry 80 used for the god-mode sweep (150 percent of Entry 79's own 200-step sweep length).")
    parser.add_argument("--torque-limit", type=float, default=7.0)
    parser.add_argument("--artificial-viscosity", type=float, default=0.1)
    parser.add_argument("--reset-velocity", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0, help="Skip checkpoints below this absolute step count.")
    parser.add_argument("--include-passive", action="store_true", help="Also run a zero-torque baseline first, to check whether this reset_velocity/torque_limit combination is survivable by ANY policy at all -- see evaluate_passive()'s own docstring for why this matters.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoints = sorted(glob.glob(args.checkpoint_glob), key=_checkpoint_step_count)
    checkpoints = [c for c in checkpoints if _checkpoint_step_count(c) >= args.min_step]

    if not checkpoints:
        print(f"No checkpoints matched {args.checkpoint_glob!r} with step >= {args.min_step}")
        return

    print(f"Sweeping {len(checkpoints)} checkpoints, {args.n_steps} steps each, seed={args.seed}")
    print(f"torque_limit={args.torque_limit} artificial_viscosity={args.artificial_viscosity} reset_velocity={args.reset_velocity}")

    if args.include_passive:
        passive = evaluate_passive(
            args.n_steps, args.torque_limit, args.artificial_viscosity,
            args.reset_velocity, args.seed,
        )
        print(f"PASSIVE (zero-torque) baseline: {passive}")
        print(
            "-> if this survives about as few steps as the checkpoints below, "
            "the flat sweep result reflects this disturbance/settings combo, "
            "not the campaign's own policy quality."
        )
        print("-" * 90)

    print(f"{'checkpoint':>10} | {'steps':>6} | {'final_ang_vel':>13} | {'final_slosh':>12} | {'max_slosh':>10} | terminated_early")
    print("-" * 90)

    results: List[Tuple[int, Dict[str, float]]] = []
    for model_path in checkpoints:
        steps = _checkpoint_step_count(model_path)
        vecnormalize_path = _vecnormalize_path_for(model_path, steps)
        metrics = evaluate_checkpoint(
            model_path, vecnormalize_path, args.n_steps, args.torque_limit,
            args.artificial_viscosity, args.reset_velocity, args.seed,
        )
        results.append((steps, metrics))
        print(
            f"{steps:>10} | {metrics['survived_steps']:>6} | {metrics['final_angular_velocity']:>13.4f} | "
            f"{metrics['final_slosh_energy']:>12.6f} | {metrics['max_slosh_energy']:>10.6f} | {metrics['terminated_early']}"
        )

    # Best = survived the longest, tie-broken by settling closest to zero
    # angular velocity -- same criterion Entry 79 used to pick 514560 over
    # checkpoints that merely "survived" while still spinning near the trip
    # boundary.
    best = max(
        results,
        key=lambda r: (r[1]["survived_steps"], -abs(r[1]["final_angular_velocity"])),
    )
    print("-" * 90)
    print(f"Best (longest survival, tie-broken by lowest final |angular_velocity|): checkpoint {best[0]} -- {best[1]}")
    final_steps = max(steps for steps, _ in results)
    if best[0] != final_steps:
        print(
            f"NOTE: best checkpoint ({best[0]}) is NOT the campaign's final checkpoint "
            f"({final_steps}) -- same 'more training isn't automatically better' pattern "
            f"already found for the god-mode campaign (Entry 79). Do not default to using "
            f"the final checkpoint without checking this table."
        )


if __name__ == "__main__":
    main()
