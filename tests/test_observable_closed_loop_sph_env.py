"""Tests for environment/observable_closed_loop_sph_env.py -- needs the
real from-source Chrono::FSI build (pychrono.fsi), not the plain
conda-forge pychrono this project's other tests get by with. Guarded with
importorskip, same pattern as tests/test_closed_loop_sph_tank.py.

Deliberately FEW tests, each doing several checks in one env construction
-- ClosedLoopSPHEnv.reset() rebuilds an entire FSI scene from scratch,
a real, non-trivial cost (see that module's own docstring); this file
avoids paying that cost once per assertion the way the fast-surrogate
tests/test_observable_sloshing_env.py can afford to.

UNTESTED as of 2026-09-16 (see observable_closed_loop_sph_env.py's own
module docstring for why) -- written carefully against
ClosedLoopSPHEnv's real, already-read structure, but not yet actually run.
"""

import numpy as np
import pytest

pytest.importorskip(
    "pychrono.fsi",
    reason="Real Chrono::FSI build (from-source, CUDA) is required for these integration tests.",
)

from config.simulation_config import EnvironmentConfig  # noqa: E402
from environment.observable_closed_loop_sph_env import ObservableClosedLoopSPHEnv  # noqa: E402


def test_observation_space_and_reset_shape_and_zeroed_history():
    env = ObservableClosedLoopSPHEnv(env_config=EnvironmentConfig(max_motor_torque_nm=7.0))
    assert env.observation_space.shape == (7,)

    obs, _info = env.reset(seed=0)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    # prev_applied_torque (index 6) must start at 0 -- no action has been
    # taken yet.
    assert obs[6] == pytest.approx(0.0)
    # Pressures (indices 2, 3) must never be negative, at rest or otherwise
    # (wall_pressures()'s own clamp) -- checked here against the REAL FSI
    # fluid state, not just the pure-math unit tests already covering
    # wall_pressures() itself in test_observable_sloshing_env.py.
    assert obs[2] >= 0.0
    assert obs[3] >= 0.0
    env.close()


def test_prev_torque_reflects_the_action_just_applied_against_real_fsi():
    env = ObservableClosedLoopSPHEnv(env_config=EnvironmentConfig(max_motor_torque_nm=7.0))
    env.reset(seed=0)

    obs, _reward, terminated, truncated, _info = env.step(np.array([0.5], dtype=np.float32))
    assert obs[6] == pytest.approx(3.5)  # 0.5 * 7.0 N*m
    assert not terminated and not truncated  # a single step shouldn't already trip a safety limit

    obs, _reward, _terminated, _truncated, _info = env.step(np.array([-1.0], dtype=np.float32))
    assert obs[6] == pytest.approx(-7.0)
    env.close()


def test_never_exposes_fluid_com_directly_against_real_fsi():
    """Same regression guard as ObservableSloshingEnv's own test, but
    against the real FluidState this class actually reads from
    ClosedLoopSPHTank.get_fluid_state() -- confirms the backend-agnostic
    FluidState interface really does behave the same way here, not just
    in the fast-surrogate case."""
    env = ObservableClosedLoopSPHEnv(env_config=EnvironmentConfig(max_motor_torque_nm=7.0))
    obs, _info = env.reset(seed=0)
    fluid_state = env._tank.get_fluid_state()
    forbidden = {
        fluid_state.com_position[0],
        fluid_state.com_position[1],
        fluid_state.com_velocity[0],
        fluid_state.com_velocity[1],
    } - {0.0}
    assert not (forbidden & set(obs.tolist()))
    env.close()
