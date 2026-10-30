"""
Phase 15 -- auto-restarting supervisor around training.train_ppo_closed_loop_sph.

Context (the development log Entries 57-58, and the follow-up crash documented
in the entry after 58): the closed-loop SPH tank can hit a real,
UNRECOVERABLE native process abort (`particle derivVel is NAN` deep inside
Chrono_fsisph.dll, an unhandled C++ exception -- not something Python's own
try/except can ever catch) even with the mid-substep angular-velocity
safety check (Entry 58) in place. That check catches most excursions (it
fired cleanly 4 times in one real run, including one where the tank
angular velocity had already reached -7.18 rad/s within a single control
step's ~200 substeps) but not all of them -- one crash was traced to a
NaN that originated in the SPH particle-force solver itself
(SphForceWCSPH.cu's RhsCFD/collideD) WITHOUT first producing an excessive
TANK angular velocity, so a check that only watches the rigid tank body's
kinematics can't see it coming.

Rather than chase an ever-more-elaborate in-process check that may never
be provably complete (SPH solver internals are not something this
project controls), this script accepts that occasional crashes are a
real, permanent characteristic of this backend and makes the TRAINING
CAMPAIGN resilient to them instead: it runs
training.train_ppo_closed_loop_sph as a CHILD PROCESS (so a native crash
only kills that child, not this supervisor), and on a non-zero exit,
finds the latest surviving checkpoint (+ its VecNormalize stats, saved
alongside every checkpoint since Entry 58's save_vecnormalize=True fix)
and relaunches from there for the remaining timestep budget -- repeating
until the target is reached or --max-restarts is hit.

Run (continuing an existing campaign toward a total of 16384 NEW
timesteps beyond wherever --base-model's own num_timesteps already was --
read directly from that model at startup, not a separately-typed number):
    python -m scripts.supervised_closed_loop_finetune \
        --base-model models/ppo_sloshing_closed_loop_sph \
        --base-vecnormalize-path models/ppo_sloshing_closed_loop_sph_vecnormalize.pkl \
        --additional-timesteps 16384 --checkpoint-dir models \
        --checkpoint-prefix ppo_sloshing_closed_loop_sph_campaign \
        --n-steps 256 --torque-limit 7.0 --artificial-viscosity 0.1 \
        --checkpoint-every-steps 512 --max-restarts 30

Requires the real local Chrono::FSI build/env, same as
train_ppo_closed_loop_sph.py itself -- this script only orchestrates
subprocess launches, it does not import pychrono anywhere itself (so it
stays runnable even to inspect --help without the env active).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_STEP_SUFFIX_RE = re.compile(r"_(\d+)_steps\.zip$")


def _find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> Optional[Tuple[str, int]]:
    """Return (path_without_.zip, step_count) for the highest-step checkpoint
    matching '<checkpoint_dir>/<prefix>_<N>_steps.zip', or None if none exist."""
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    best: Optional[Tuple[str, int]] = None
    for path in glob.glob(pattern):
        match = _STEP_SUFFIX_RE.search(os.path.basename(path))
        if not match:
            continue
        steps = int(match.group(1))
        if best is None or steps > best[1]:
            best = (path[: -len(".zip")], steps)
    return best


def _vecnormalize_path_for(checkpoint_path_no_ext: str, checkpoint_prefix: str, checkpoint_dir: str, steps: int) -> Optional[str]:
    """CheckpointCallback(save_vecnormalize=True) names these
    '<prefix>_vecnormalize_<N>_steps.pkl', not '<checkpoint>_vecnormalize.pkl'
    -- a different naming convention than train_ppo_closed_loop_sph.py's own
    final save, so this can't just string-substitute the checkpoint path."""
    candidate = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_vecnormalize_{steps}_steps.pkl")
    return candidate if os.path.exists(candidate) else None


def _run_one_attempt(
    base_model: str, base_vecnormalize_path: Optional[str], total_timesteps: int,
    output_prefix: str, n_steps: int, batch_size: int, torque_limit: float,
    reset_velocity: float, artificial_viscosity: float, checkpoint_every_steps: int,
    checkpoint_dir: str, seed: int, training_module: str, boundary_method: str = "adami",
) -> int:
    """Launch `training_module` (a module implementing the same
    --base-model/--total-timesteps/... CLI as training.train_ppo_closed_loop_sph
    -- training.train_ppo_observable_closed_loop_sph is the other real
    implementer, added 2026-09-16) as a child process. Returns its exit code
    (0 = the full total_timesteps for THIS attempt completed; non-zero,
    including a native-crash-induced negative/abort code = it died early --
    checkpoints already written before the crash still survive on disk
    regardless)."""
    cmd = [
        sys.executable, "-m", training_module,
        "--base-model", base_model,
        "--total-timesteps", str(total_timesteps),
        "--n-steps", str(n_steps),
        "--batch-size", str(batch_size),
        "--torque-limit", str(torque_limit),
        "--reset-velocity", str(reset_velocity),
        "--artificial-viscosity", str(artificial_viscosity),
        "--boundary-method", boundary_method,
        "--checkpoint-every-steps", str(checkpoint_every_steps),
        "--output", os.path.join(checkpoint_dir, output_prefix),
        "--seed", str(seed),
    ]
    if base_vecnormalize_path is not None:
        cmd += ["--base-vecnormalize-path", base_vecnormalize_path]

    logger.info("Launching attempt: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def supervise(args: argparse.Namespace) -> None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # The campaign's true starting point is whatever --base-model's OWN
    # num_timesteps already is -- read it directly from the model rather
    # than trusting a separately-typed CLI number (a real bug caught before
    # this script's first real run: --base-model can itself already be a
    # mid-campaign checkpoint with more timesteps than some earlier "the
    # original base model had N" assumption would account for).
    from stable_baselines3 import PPO
    campaign_start_num_timesteps = PPO.load(args.base_model).num_timesteps
    logger.info(
        "Campaign baseline: %s has num_timesteps=%d -- --additional-timesteps "
        "counts NEW steps beyond this point.", args.base_model, campaign_start_num_timesteps,
    )

    latest = _find_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
    if latest is None:
        current_base_model = args.base_model
        current_vecnormalize = args.base_vecnormalize_path
        completed_new_steps = 0
        logger.info(
            "No existing campaign checkpoint found under prefix %r -- starting "
            "fresh from %s.", args.checkpoint_prefix, args.base_model,
        )
    else:
        current_base_model, steps_at_latest = latest
        current_vecnormalize = _vecnormalize_path_for(
            current_base_model, args.checkpoint_prefix, args.checkpoint_dir, steps_at_latest
        )
        # steps_at_latest is this checkpoint's ABSOLUTE SB3 num_timesteps
        # (embedded via model.save()/CheckpointCallback), not a delta --
        # recover how many NEW timesteps this campaign has done so far by
        # diffing against the campaign's own starting point.
        completed_new_steps = steps_at_latest - campaign_start_num_timesteps
        logger.info(
            "Resuming existing campaign from %s (%d absolute timesteps, "
            "%d new timesteps completed so far). VecNormalize: %s",
            current_base_model, steps_at_latest, completed_new_steps,
            current_vecnormalize or "NOT FOUND -- will start fresh stats",
        )

    restart_count = 0
    consecutive_no_progress = 0
    while completed_new_steps < args.additional_timesteps:
        if restart_count >= args.max_restarts:
            logger.error(
                "Hit --max-restarts (%d) with %d/%d new timesteps completed -- "
                "stopping. This many crashes this close together suggests a "
                "real, not-yet-understood problem, not bad luck -- investigate "
                "before just raising the limit and retrying.",
                args.max_restarts, completed_new_steps, args.additional_timesteps,
            )
            break
        if consecutive_no_progress >= args.max_consecutive_no_progress:
            logger.error(
                "%d consecutive attempts made ZERO new progress (each crashed "
                "before its own first --checkpoint-every-steps boundary) -- "
                "stopping well short of --max-restarts (%d used of %d). A real "
                "physics-driven crash's timing varies attempt to attempt (this "
                "campaign has seen crashes anywhere from ~200 to ~1500 steps "
                "in); the SAME failure-before-any-checkpoint outcome repeating "
                "%d times in a row is a much stronger signal of an actual code "
                "or config bug than of bad luck -- investigate the most recent "
                "attempt's own output before blindly raising this limit and "
                "retrying.",
                consecutive_no_progress, restart_count, args.max_restarts,
                consecutive_no_progress,
            )
            break

        remaining = args.additional_timesteps - completed_new_steps
        attempt_start = time.time()
        exit_code = _run_one_attempt(
            base_model=current_base_model,
            base_vecnormalize_path=current_vecnormalize,
            total_timesteps=remaining,
            output_prefix=args.checkpoint_prefix,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            torque_limit=args.torque_limit,
            reset_velocity=args.reset_velocity,
            artificial_viscosity=args.artificial_viscosity,
            checkpoint_every_steps=args.checkpoint_every_steps,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed + restart_count,  # vary the seed across restarts
            training_module=args.training_module,
            boundary_method=args.boundary_method,
        )
        elapsed = time.time() - attempt_start

        # A crash can happen before this attempt's FIRST checkpoint boundary
        # (--checkpoint-every-steps) -- confirmed for real: one attempt ran
        # a full episode plus ~230 more steps (real simulated progress) but
        # crashed at ~408 collected timesteps, short of the 512-step
        # checkpoint boundary, so PPO's CheckpointCallback never got to
        # write anything. That in-memory progress is genuinely
        # unrecoverable (PPO does not checkpoint mid-rollout) -- but it is
        # NOT a reason to give up on the whole campaign, only to retry from
        # the same base model this attempt itself started from.
        latest = _find_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)
        if latest is None:
            made_progress = False
            # current_base_model / current_vecnormalize are left exactly as
            # they were going into this attempt -- the next attempt just
            # retries from the same starting point.
        else:
            new_base_model, steps_at_latest = latest
            new_completed = steps_at_latest - campaign_start_num_timesteps
            made_progress = new_completed > completed_new_steps
            current_base_model = new_base_model
            current_vecnormalize = _vecnormalize_path_for(
                current_base_model, args.checkpoint_prefix, args.checkpoint_dir, steps_at_latest
            )
            completed_new_steps = new_completed

        consecutive_no_progress = 0 if made_progress else consecutive_no_progress + 1

        if exit_code == 0:
            logger.info(
                "Attempt completed cleanly (%.1fs). Total new timesteps: %d/%d.",
                elapsed, completed_new_steps, args.additional_timesteps,
            )
        else:
            logger.warning(
                "Attempt exited with code %s after %.1fs (likely a native SPH "
                "solver crash -- see this script's own docstring). Progress "
                "this attempt: %s. Total new timesteps: %d/%d. Resuming from "
                "%s.",
                exit_code, elapsed,
                "made progress" if made_progress else "NONE (crashed before this attempt's first checkpoint boundary -- retrying from the same starting point)",
                completed_new_steps, args.additional_timesteps, current_base_model,
            )
            restart_count += 1

    logger.info(
        "Supervisor finished: %d/%d new timesteps completed across %d restart(s). "
        "Latest checkpoint: %s",
        completed_new_steps, args.additional_timesteps, restart_count, current_base_model,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-module", default="training.train_ppo_closed_loop_sph", help="Which training module to launch each attempt with -- must implement the same CLI (--base-model, --total-timesteps, --n-steps, --batch-size, --torque-limit, --reset-velocity, --artificial-viscosity, --boundary-method, --checkpoint-every-steps, --output, --seed, --base-vecnormalize-path). Default is the god-mode (raw fluid-COM observation) trainer; pass training.train_ppo_observable_closed_loop_sph for the hardware-deployable pressure-observation variant (added 2026-09-16).")
    parser.add_argument("--base-model", default="models/ppo_sloshing_closed_loop_sph")
    parser.add_argument("--base-vecnormalize-path", default=None)
    parser.add_argument("--additional-timesteps", type=int, required=True, help="Total NEW timesteps this campaign should reach, on top of --base-model-num-timesteps -- NOT an absolute SB3 counter value.")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--checkpoint-prefix", required=True, help="Distinct name for this campaign's checkpoints (e.g. 'ppo_sloshing_closed_loop_sph_16k_campaign') -- used both as CheckpointCallback's name_prefix and to find the latest checkpoint to resume from on restart.")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torque-limit", type=float, default=7.0)
    parser.add_argument("--reset-velocity", type=float, default=0.05)
    parser.add_argument("--artificial-viscosity", type=float, default=0.1)
    parser.add_argument("--boundary-method", choices=["adami", "holmes"], default="adami", help="Forwarded to the child training module's own --boundary-method (the development log Entry 119) -- 'adami' matches every prior campaign's dynamics unchanged.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=512)
    parser.add_argument("--max-restarts", type=int, default=10, help="Hard cap on total restarts -- kept deliberately small, not generous: this campaign HAS seen real crashes (physics-driven, not a bug) with hours between them, so a large cap risks silently burning a long time on a run that turns out to be stuck for a genuine code/config reason. Re-invoke with a fresh budget once you've confirmed real progress is happening, rather than setting this high up front.")
    parser.add_argument("--max-consecutive-no-progress", type=int, default=3, help="Stop even before --max-restarts if this many attempts IN A ROW each crash before their own first checkpoint boundary (zero new progress) -- a real physics-driven crash's timing varies attempt to attempt, so the same immediate-failure outcome repeating is a stronger signal of an actual bug than of bad luck.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    supervise(_parse_args())
