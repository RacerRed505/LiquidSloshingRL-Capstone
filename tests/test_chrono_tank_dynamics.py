"""
Integration tests for Phase 10: builders/motor_builder.py's MotorKind.TORQUE
support, and environment/chrono_tank_dynamics.py::ChronoTankDynamics.

These require the REAL PyChrono bindings:
    conda install -c conda-forge pychrono

Guarded with `pytest.importorskip`, same pattern as
test_pivot_and_motor.py / test_world_and_tank.py -- automatically SKIPPED
(not failed) anywhere the real compiled bindings aren't installed,
including the assistant's own sandbox. Run these on your machine.

Run:
    pytest tests/test_chrono_tank_dynamics.py -v
"""

import pytest

chrono = pytest.importorskip(
    "pychrono.core",
    reason="Real PyChrono (conda-forge) is required for these integration tests.",
)

from builders.motor_builder import MotorBuilder
from builders.pivot_builder import PivotBuilder
from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import (
    EnvironmentConfig,
    MotorConfig,
    MotorKind,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.chrono_tank_dynamics import ChronoTankDynamics


def _build_torque_driven_tank():
    system = WorldBuilder(WorldConfig()).build()
    tank_body = TankBuilder(TankConfig(fixed=False)).build(system)
    pivot_body = PivotBuilder(PivotConfig()).build(system)
    motor = MotorBuilder(PivotConfig(), MotorConfig(kind=MotorKind.TORQUE)).build(
        system, pivot_body, tank_body
    )
    return system, tank_body, pivot_body, motor


# -- builders/motor_builder.py MotorKind.TORQUE ---------------------------


def test_torque_motor_no_longer_raises_not_implemented():
    # This used to raise NotImplementedError before Phase 10 -- the
    # assertion here is just that construction succeeds.
    MotorBuilder(PivotConfig(), MotorConfig(kind=MotorKind.TORQUE))


def test_torque_motor_builder_adds_exactly_one_link():
    system, _, _, motor = _build_torque_driven_tank()
    assert len(system.GetLinks()) == 1
    assert isinstance(motor, chrono.ChLinkMotorRotationTorque)


def test_torque_motor_starts_at_zero_angle_and_zero_commanded_torque():
    _, _, _, motor = _build_torque_driven_tank()
    angle, angular_velocity = MotorBuilder.read_tank_motion(motor)
    assert angle == pytest.approx(0.0, abs=1e-6)
    assert angular_velocity == pytest.approx(0.0, abs=1e-6)


def test_zero_commanded_torque_leaves_tank_essentially_at_rest():
    system, _, _, motor = _build_torque_driven_tank()
    MotorBuilder.set_commanded_torque(motor, 0.0)
    dt = 1.0e-3
    for _ in range(200):
        system.DoStepDynamics(dt)
    angle, _ = MotorBuilder.read_tank_motion(motor)
    # No restoring/disturbing force is acting (gravity alone, tank pivot
    # at its own footprint center) -- allow a small numerical-drift
    # tolerance rather than requiring bit-exact zero.
    assert angle == pytest.approx(0.0, abs=1e-3)


def test_positive_commanded_torque_rotates_tank_in_positive_direction():
    system, _, _, motor = _build_torque_driven_tank()
    MotorBuilder.set_commanded_torque(motor, 1.0)
    dt = 1.0e-3
    for _ in range(200):
        system.DoStepDynamics(dt)
    angle, angular_velocity = MotorBuilder.read_tank_motion(motor)
    assert angle > 0.0
    assert angular_velocity > 0.0


def test_negative_commanded_torque_rotates_tank_in_negative_direction():
    system, _, _, motor = _build_torque_driven_tank()
    MotorBuilder.set_commanded_torque(motor, -1.0)
    dt = 1.0e-3
    for _ in range(200):
        system.DoStepDynamics(dt)
    angle, angular_velocity = MotorBuilder.read_tank_motion(motor)
    assert angle < 0.0
    assert angular_velocity < 0.0


def test_set_commanded_torque_can_change_value_mid_simulation():
    """Confirms the mutable-ChFunction technique actually takes effect on
    the NEXT step, not just at motor-construction time -- this is the
    exact mechanism environment/chrono_tank_dynamics.py's step() depends
    on being called once per control step."""
    system, _, _, motor = _build_torque_driven_tank()
    dt = 1.0e-3

    MotorBuilder.set_commanded_torque(motor, 1.0)
    for _ in range(50):
        system.DoStepDynamics(dt)
    angle_after_positive, velocity_after_positive = MotorBuilder.read_tank_motion(motor)
    assert velocity_after_positive > 0.0

    MotorBuilder.set_commanded_torque(motor, -5.0)
    for _ in range(200):
        system.DoStepDynamics(dt)
    _, velocity_after_negative = MotorBuilder.read_tank_motion(motor)
    # A strong reversed torque, held long enough, must eventually reverse
    # the angular velocity's sign -- if set_commanded_torque were a no-op
    # after the first call, this would stay positive.
    assert velocity_after_negative < velocity_after_positive


# -- environment/chrono_tank_dynamics.py::ChronoTankDynamics --------------


def test_rejects_fixed_tank_config():
    with pytest.raises(ValueError):
        ChronoTankDynamics(tank_config=TankConfig(fixed=True))


def test_starts_at_rest():
    dynamics = ChronoTankDynamics()
    assert dynamics.angle == pytest.approx(0.0, abs=1e-6)
    assert dynamics.angular_velocity == pytest.approx(0.0, abs=1e-6)


def test_torque_limit_matches_env_config():
    env_config = EnvironmentConfig(max_motor_torque_nm=3.5)
    dynamics = ChronoTankDynamics(env_config=env_config)
    assert dynamics.torque_limit == pytest.approx(3.5)


def test_step_with_positive_torque_increases_angle():
    dynamics = ChronoTankDynamics()
    dt = 1.0e-3
    for _ in range(200):
        dynamics.step(dt, commanded_torque=1.0)
    assert dynamics.angle > 0.0
    assert dynamics.angular_velocity > 0.0


def test_commanded_torque_beyond_limit_is_clipped():
    """Stepping with an absurdly large torque should produce the SAME
    motion as stepping at exactly torque_limit -- proving the clip in
    ChronoTankDynamics.step() actually reaches the real motor, the same
    behavioral contract TankDynamicsModel.step() already guarantees."""
    env_config = EnvironmentConfig(max_motor_torque_nm=1.0)
    dt = 1.0e-3
    n_steps = 100

    at_limit = ChronoTankDynamics(env_config=env_config)
    for _ in range(n_steps):
        at_limit.step(dt, commanded_torque=1.0)

    way_over_limit = ChronoTankDynamics(env_config=env_config)
    for _ in range(n_steps):
        way_over_limit.step(dt, commanded_torque=1000.0)

    assert way_over_limit.angle == pytest.approx(at_limit.angle, rel=1e-6, abs=1e-9)
    assert way_over_limit.angular_velocity == pytest.approx(
        at_limit.angular_velocity, rel=1e-6, abs=1e-9
    )


def test_reset_returns_to_rest_after_motion():
    dynamics = ChronoTankDynamics()
    dt = 1.0e-3
    for _ in range(200):
        dynamics.step(dt, commanded_torque=1.0)
    assert dynamics.angle != pytest.approx(0.0, abs=1e-6)  # sanity: it did move

    dynamics.reset()
    assert dynamics.angle == pytest.approx(0.0, abs=1e-6)
    assert dynamics.angular_velocity == pytest.approx(0.0, abs=1e-6)


def test_exposes_the_same_surface_as_the_analytic_surrogate():
    """Duck-typing contract check: environment/sloshing_env.py only ever
    calls these four things on whatever `tank_dynamics` it was given."""
    dynamics = ChronoTankDynamics()
    assert hasattr(dynamics, "reset") and callable(dynamics.reset)
    assert hasattr(dynamics, "step") and callable(dynamics.step)
    assert isinstance(dynamics.angle, float)
    assert isinstance(dynamics.angular_velocity, float)
