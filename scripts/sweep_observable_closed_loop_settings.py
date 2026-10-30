"""
Phase 15 -- bounded torque-limit/reset-velocity settings sweep for the
observable closed-loop SPH campaign.

Follows directly from the development log Entry 84's finding: at the training
settings (torque_limit=7.0, reset_velocity=0.05), a zero-torque passive
baseline trips the safety limit at almost the same step as the fully
fine-tuned final checkpoint -- meaning that specific disturbance/torque
combination isn't survivable by ANY policy, trained or not. This script
asks a narrower, bounded question instead of retraining anything: is
there an EASIER settings combination (milder disturbance and/or more
torque authority) where the campaign's ALREADY-TRAINED checkpoint shows a
real, meaningful advantage over doing nothing? If yes, that's useful
evidence the policy did learn something real, just not enough to handle
the specific disturbance magnitude it was trained and evaluated against.
If no combination shows a real advantage anywhere in the grid, that's a
stronger version of Entry 84's negative finding.

Deliberately reuses evaluate_checkpoint()/evaluate_passive() from
scripts.sweep_observable_closed_loop_checkpoints UNCHANGED, so this can't
drift from that script's own real env construction -- this file only adds
the settings-grid loop on top.

Needs the real FSI/CUDA build -- run on Colab or the second RTX machine,
not this project's original dev machine.

Run (small, fixed 3x3 grid by default -- NOT an open-ended search):
    python -m scripts.sweep_observable_closed_loop_settings \
        --model-path models/ppo_sloshing_observable_closed_loop_sph_campaign_524288_steps.zip \
        --n-steps 250
"""

from __future__ import annotations

import argparse
import itertools
from typing import List

from scripts.sweep_observable_closed_loop_checkpoints import (
    _checkpoint_step_count,
    _vecnormalize_path_for,
    evaluate_checkpoint,
    evaluate_passive,
)

# Deliberately small and fixed -- this is a bounded sanity check, not a
# hyperparameter search. torque_limit=7.0/reset_velocity=0.05 is the
# training-time point (already known, from Entry 84, to be unsurvivable by
# anything); the rest of the grid tests progressively easier combinations.
DEFAULT_TORQUE_LIMITS = [7.0, 10.0, 14.0]
DEFAULT_RESET_VELOCITIES = [0.05, 0.03, 0.02]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="models/ppo_sloshing_observable_closed_loop_sph_campaign_524288_steps.zip",
    )
    parser.add_argument("--vecnormalize-path", default=None, help="Defaults to the matching '_vecnormalize_<N>_steps.pkl' file next to --model-path.")
    parser.add_argument("--n-steps", type=int, default=250)
    parser.add_argument("--artificial-viscosity", type=float, default=0.1)
    parser.add_argument("--torque-limits", type=float, nargs="+", default=DEFAULT_TORQUE_LIMITS)
    parser.add_argument("--reset-velocities", type=float, nargs="+", default=DEFAULT_RESET_VELOCITIES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        steps = _checkpoint_step_count(args.model_path)
        vecnormalize_path = (
            _vecnormalize_path_for(args.model_path, steps) if steps >= 0 else ""
        )

    combos = list(itertools.product(args.torque_limits, args.reset_velocities))
    print(f"Settings sweep for {args.model_path} ({len(combos)} combinations, {args.n_steps}-step cap each)")
    print(
        f"{'torque_limit':>12} | {'reset_vel':>9} | {'passive_steps':>13} | "
        f"{'policy_steps':>12} | {'advantage':>9} | {'policy_final_slosh':>18}"
    )
    print("-" * 90)

    results: List[dict] = []
    for torque_limit, reset_velocity in combos:
        passive = evaluate_passive(
            args.n_steps, torque_limit, args.artificial_viscosity, reset_velocity, args.seed
        )
        policy = evaluate_checkpoint(
            args.model_path, vecnormalize_path, args.n_steps, torque_limit,
            args.artificial_viscosity, reset_velocity, args.seed,
        )
        advantage = policy["survived_steps"] - passive["survived_steps"]
        row = {
            "torque_limit": torque_limit,
            "reset_velocity": reset_velocity,
            "passive_steps": passive["survived_steps"],
            "policy_steps": policy["survived_steps"],
            "advantage": advantage,
            "policy_final_slosh": policy["final_slosh_energy"],
        }
        results.append(row)
        print(
            f"{torque_limit:>12.1f} | {reset_velocity:>9.3f} | {passive['survived_steps']:>13} | "
            f"{policy['survived_steps']:>12} | {advantage:>9} | {policy['final_slosh_energy']:>18.6f}"
        )

    print("-" * 90)
    best = max(results, key=lambda r: r["advantage"])
    print(f"Largest policy-vs-passive advantage: torque_limit={best['torque_limit']}, reset_velocity={best['reset_velocity']} -> +{best['advantage']} steps")
    if best["advantage"] <= 2:
        print(
            "No combination in this grid shows a meaningful advantage over the passive "
            "baseline -- a stronger version of Entry 84's negative finding, not just an "
            "artifact of the original training-time settings."
        )


if __name__ == "__main__":
    main()
