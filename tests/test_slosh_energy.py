"""Tests for the kinetic/potential/total slosh-energy split (second-review
item: "convective/sloshing energy -- check if valid")."""

import math

import pytest

from physics.backend_interface import FluidState
from physics.pendulum_backend import PendulumBackend
from scripts.recheck_slosh_energy import pendulum_energy_split, sph_fluid_mass


def _backend(damping: float = 0.0) -> PendulumBackend:
    return PendulumBackend(pendulum_length=0.14, equivalent_mass=9.7, damping_coefficient=damping)


def test_potential_energy_matches_closed_form():
    backend = _backend()
    backend._theta = 0.2
    state = backend.get_state()
    assert state.slosh_potential_energy == pytest.approx(9.7 * 9.81 * 0.14 * (1.0 - math.cos(0.2)))


def test_at_rest_all_energy_terms_are_zero():
    state = _backend().get_state()
    assert state.slosh_energy == 0.0
    assert state.slosh_potential_energy == 0.0
    assert state.slosh_total_energy == 0.0


def test_undamped_free_swing_conserves_total_but_not_kinetic_energy():
    """The kinetic-only value swings to ~0 at every turning point; the
    total should stay (nearly) constant for an undamped, unforced pendulum."""
    backend = _backend(damping=0.0)
    backend.apply_disturbance((0.5, 0.0, 0.0))
    kinetic, total = [], []
    for _ in range(4000):
        backend.step(1e-4, (0.0, 0.0))
        state = backend.get_state()
        kinetic.append(state.slosh_energy)
        total.append(state.slosh_total_energy)
    assert min(kinetic) < 0.05 * max(kinetic)
    assert (max(total) - min(total)) / max(total) < 0.01


def test_total_energy_is_none_when_backend_reports_no_potential():
    state = FluidState(com_position=(0.0, 0.0, 0.0), com_velocity=(1.0, 0.0, 0.0), slosh_energy=0.5)
    assert state.slosh_potential_energy is None
    assert state.slosh_total_energy is None


def test_offline_recheck_recovers_potential_from_saved_com_height():
    mass, length, g = 9.7, 0.14, 9.81
    theta = 0.3
    rows = [{"pendulum_slosh_energy": "0.0", "pendulum_fluid_com_y": str(-length * math.cos(theta))}]
    split = pendulum_energy_split(rows, mass, length, g)
    assert split["potential"][0] == pytest.approx(mass * g * length * (1.0 - math.cos(theta)))


def test_offline_recheck_recovers_sph_fluid_mass():
    rows = [
        {"sph_fluid_com_vx": "0.0", "sph_fluid_com_vy": "0.0", "sph_slosh_energy": "0.0"},
        {"sph_fluid_com_vx": "0.3", "sph_fluid_com_vy": "0.4", "sph_slosh_energy": str(0.5 * 18.0 * 0.25)},
    ]
    assert sph_fluid_mass(rows) == pytest.approx(18.0)
