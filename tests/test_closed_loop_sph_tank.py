"""
Tests for physics/closed_loop_sph_tank.py (Phase 15).

Requires the real Chrono::FSI Python bindings (pychrono.fsi), same as
tests/test_sph_backend.py -- NOT available in this project's local conda
env (confirmed, the development log Entry 19). Guarded with
pytest.importorskip, same pattern as test_sph_backend.py -- SKIPPED
(not failed) here, meant to actually run on the RTX 3090-class machine
this module was built for (or Colab, as a fallback), after the ctypes
preload sequence documented in the project notes' Phase 13-14 section.

Run (on the real FSI-capable machine, after the preload):
    pytest tests/test_closed_loop_sph_tank.py -v
"""

import pytest

pytest.importorskip(
    "pychrono.fsi",
    reason="Real Chrono::FSI Python bindings are required for these tests -- "
    "see the project notes' Phase 13-14 section for the build + ctypes preload "
    "sequence needed before this import will succeed.",
)

from config.simulation_config import FluidConfig, TankConfig  # noqa: E402
from physics.backend_interface import FluidState  # noqa: E402
from physics.closed_loop_sph_tank import ClosedLoopSPHTank  # noqa: E402


def test_closed_loop_tank_builds_and_reports_state():
    tank = ClosedLoopSPHTank()
    assert isinstance(tank.angle, float)
    assert isinstance(tank.angular_velocity, float)
    state = tank.get_fluid_state()
    assert isinstance(state, FluidState)


def test_closed_loop_tank_torque_and_step_run_without_crashing():
    tank = ClosedLoopSPHTank()
    tank.set_commanded_torque(1.0)
    tank.step(0.02)
    state = tank.get_fluid_state()
    assert state.slosh_energy >= 0.0


def test_closed_loop_tank_torque_is_clipped():
    tank = ClosedLoopSPHTank(env_config=None)
    tank.set_commanded_torque(1000.0)
    assert tank._torque_function.value == tank._torque_limit
    tank.set_commanded_torque(-1000.0)
    assert tank._torque_function.value == -tank._torque_limit


def test_closed_loop_tank_reset_rebuilds_cleanly():
    tank = ClosedLoopSPHTank(tank_config=TankConfig(fixed=False), fluid_config=FluidConfig())
    tank.reset()
    assert isinstance(tank.get_fluid_state(), FluidState)
