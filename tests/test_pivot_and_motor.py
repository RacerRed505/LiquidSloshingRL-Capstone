"""
Integration tests for builders/pivot_builder.py and builders/motor_builder.py.

These require the REAL PyChrono bindings:
    conda install -c conda-forge pychrono

This file is guarded with `pytest.importorskip`, same pattern as
test_world_and_tank.py, so it is automatically SKIPPED (not failed) in any
environment -- including this sandbox -- where only the unrelated pip
package named "pychrono" might exist rather than the real compiled
bindings. Run these locally, in your project's real PyChrono environment,
as the Layer 2 integration check from Module 1.

Run:
    pytest tests/test_pivot_and_motor.py -v
"""

import math

import pytest

chrono = pytest.importorskip(
    "pychrono.core",
    reason="Real PyChrono (conda-forge) is required for these integration tests.",
)

from builders.motor_builder import MotorBuilder
from builders.pivot_builder import PivotBuilder
from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import MotorConfig, MotorKind, PivotConfig, TankConfig, WorldConfig


def _build_rocking_tank():
    system = WorldBuilder(WorldConfig()).build()
    tank_body = TankBuilder(TankConfig(fixed=False)).build(system)
    pivot_body = PivotBuilder(PivotConfig()).build(system)
    motor = MotorBuilder(PivotConfig(), MotorConfig()).build(system, pivot_body, tank_body)
    return system, tank_body, pivot_body, motor


def test_pivot_builder_adds_exactly_one_fixed_body():
    system = WorldBuilder(WorldConfig()).build()
    assert len(system.GetBodies()) == 0
    pivot_body = PivotBuilder(PivotConfig()).build(system)
    assert len(system.GetBodies()) == 1
    assert pivot_body.IsFixed() is True


def test_pivot_rejects_zero_axis():
    with pytest.raises(ValueError):
        PivotConfig(axis=(0.0, 0.0, 0.0))


def test_pivot_rejects_degenerate_support_size():
    with pytest.raises(ValueError):
        PivotConfig(support_size=(0.0, 0.05, 0.05))


def test_joint_frame_z_axis_matches_configured_axis():
    """The joint frame's local Z, rotated into the parent frame, must equal
    PivotConfig.axis (normalized) -- this is the property the motor's
    correctness actually depends on (Chrono spins the joint about the
    frame's own Z), so it's checked directly against a real ChFramed
    rather than trusting utilities/quaternion_utils.py's own unit tests
    alone."""
    config = PivotConfig(axis=(0.0, 0.0, 1.0))
    frame = PivotBuilder(config).joint_frame()
    local_z = chrono.ChVector3d(0, 0, 1)
    world_z = frame.TransformDirectionLocalToParent(local_z)
    assert world_z.x == pytest.approx(0.0, abs=1e-9)
    assert world_z.y == pytest.approx(0.0, abs=1e-9)
    assert world_z.z == pytest.approx(1.0, abs=1e-9)


def test_motor_rejects_unimplemented_kind():
    with pytest.raises(NotImplementedError):
        MotorBuilder(PivotConfig(), MotorConfig(kind=MotorKind.SPEED))


def test_motor_builder_adds_exactly_one_link():
    system, tank_body, pivot_body, motor = _build_rocking_tank()
    assert len(system.GetLinks()) == 1
    assert isinstance(motor, chrono.ChLinkMotorRotationAngle)


def test_motor_prescribes_zero_angle_at_t_zero():
    _, _, _, motor = _build_rocking_tank()
    angle, _ = MotorBuilder.read_tank_motion(motor)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_motor_angle_tracks_sine_profile_after_stepping():
    """End-to-end: step the real system forward and confirm the motor's
    reported angle matches MotorConfig's own amplitude/frequency, not just
    that *something* moves."""
    system, tank_body, pivot_body, motor = _build_rocking_tank()
    motor_config = MotorConfig()
    dt = 1.0e-3
    t = 0.0
    for _ in range(50):
        system.DoStepDynamics(dt)
        t += dt

    angle, _ = MotorBuilder.read_tank_motion(motor)
    expected = motor_config.amplitude_rad * math.sin(2.0 * math.pi * motor_config.frequency_hz * t)
    assert angle == pytest.approx(expected, abs=1e-3)


def test_tank_is_no_longer_kinematically_fixed_once_motor_driven():
    """Phase 6 replaces TankConfig.fixed=True with the motor joint -- this
    just confirms the demo actually constructs the tank with fixed=False,
    per TankConfig's own docstring note about when that flips."""
    _, tank_body, _, _ = _build_rocking_tank()
    assert tank_body.IsFixed() is False
