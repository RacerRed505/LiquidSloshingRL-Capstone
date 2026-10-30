"""
Integration tests for builders/world_builder.py and builders/tank_builder.py.

These require the REAL PyChrono bindings:
    conda install -c conda-forge pychrono

This file is guarded with `pytest.importorskip`, so it is automatically
SKIPPED (not failed) in any environment -- including this sandbox -- where
only the unrelated pip package named "pychrono" might exist rather than
the real compiled bindings. Run these locally, in your project's real
PyChrono environment, as the Layer 2 integration check from Module 1.

Run:
    pytest tests/test_world_and_tank.py -v
"""

import pytest

chrono = pytest.importorskip(
    "pychrono.core",
    reason="Real PyChrono (conda-forge) is required for these integration tests.",
)

from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import TankConfig, WorldConfig


def test_world_builder_sets_gravity():
    system = WorldBuilder(WorldConfig(gravity=(0.0, -9.81, 0.0))).build()
    g = system.GetGravitationalAcceleration()
    assert g.y == pytest.approx(-9.81)


def test_world_builder_creates_smc_system_by_default():
    system = WorldBuilder(WorldConfig()).build()
    assert isinstance(system, chrono.ChSystemSMC)


def test_tank_builder_adds_exactly_one_body_to_system():
    system = WorldBuilder(WorldConfig()).build()
    assert len(system.GetBodies()) == 0
    TankBuilder(TankConfig()).build(system)
    assert len(system.GetBodies()) == 1


def test_tank_mass_matches_analytical_value():
    tank_config = TankConfig()
    system = WorldBuilder(WorldConfig()).build()
    body = TankBuilder(tank_config).build(system)

    expected_mass = TankBuilder(tank_config).compute_mass_properties().mass
    assert body.GetMass() == pytest.approx(expected_mass, rel=1e-6)


def test_tank_is_fixed_by_default():
    system = WorldBuilder(WorldConfig()).build()
    body = TankBuilder(TankConfig()).build(system)
    assert body.IsFixed() is True


def test_tank_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        TankConfig(length=0.01, width=0.01, wall_thickness=0.01)
