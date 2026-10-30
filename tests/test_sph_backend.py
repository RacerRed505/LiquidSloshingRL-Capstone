"""
Tests for physics/sph_backend.py (Phase 13-14).

Requires the real Chrono::FSI Python bindings (pychrono.fsi), which are
NOT available in this project's local conda env (confirmed,
the development log Entry 19: ModuleNotFoundError) -- they only exist on the
Colab GPU build described in the project notes' Phase 13-14 section, after the
ctypes.CDLL(RTLD_GLOBAL) preload sequence documented there has been run.
Guarded with pytest.importorskip, same pattern as
test_chrono_tank_dynamics.py / test_pivot_and_motor.py -- automatically
SKIPPED (not failed) anywhere pychrono.fsi isn't importable, including
this project's own local machine and any CI. pytest.importorskip alone
does not perform the ctypes preload -- these tests are only meaningful
run in a session where that preload has already happened before pytest
starts.

Run (on Colab / a future FSI-capable machine, after the preload):
    pytest tests/test_sph_backend.py -v
"""

import pytest

pytest.importorskip(
    "pychrono.fsi",
    reason="Real Chrono::FSI Python bindings are required for these tests -- "
    "see the project notes' Phase 13-14 section for the Colab build + ctypes "
    "preload sequence needed before this import will succeed.",
)

from config.simulation_config import FluidConfig, TankConfig  # noqa: E402
from physics.backend_interface import FluidState  # noqa: E402
from physics.sph_backend import SPHBackend  # noqa: E402


def test_sph_backend_satisfies_fluid_backend_interface():
    backend = SPHBackend()
    assert backend.name == "sph"
    state = backend.get_state()
    assert isinstance(state, FluidState)
    assert state.com_position[2] == 0.0
    assert state.com_velocity[2] == 0.0


def test_sph_backend_step_and_reset_run_without_crashing():
    backend = SPHBackend()
    backend.step(1.0e-3, (0.05, 0.1))
    state = backend.get_state()
    assert isinstance(state.slosh_energy, float)
    assert state.slosh_energy >= 0.0
    backend.reset()
    state_after_reset = backend.get_state()
    assert state_after_reset.com_position == (0.0, 0.0, 0.0)


def test_sph_backend_apply_disturbance_is_explicitly_not_implemented():
    """Documents the design decision (the development log Entry 41) rather than
    silently no-op-ing -- disturbance injection is out of scope for this
    validation-only backend."""
    backend = SPHBackend()
    with pytest.raises(NotImplementedError):
        backend.apply_disturbance((0.0, 0.0, 0.0))


def test_sph_backend_accepts_custom_configs():
    backend = SPHBackend(tank_config=TankConfig(fixed=False), fluid_config=FluidConfig())
    assert backend.name == "sph"
