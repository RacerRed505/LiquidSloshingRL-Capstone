"""
Phase 8 -- Torque-driven surrogate tank rotational dynamics.

environment/sloshing_env.py needs *something* to convert the RL agent's
commanded torque into the (tank_angle, tank_angular_velocity) pair that
physics/backend_interface.py::FluidBackend.step() expects as its
`tank_motion` argument. The real answer -- Phase 6's ChLinkMotorRotation
joint driving an actual ChBody -- requires pychrono, which the
environment layer is deliberately not allowed to import (see
config/simulation_config.py::EnvironmentConfig's own docstring, and
physics/backend_interface.py's docstring for the same rule applied to the
fluid side).

So this module is the tank-side equivalent of PendulumBackend: a small,
analytically-derived, PyChrono-free single-degree-of-freedom rotational
model, good enough to train an initial policy against, that Phase 10+ can
swap out for a real ChSystem-backed adapter without environment/sloshing_env.py
needing to change -- provided the replacement exposes the same reset/step/
angle/angular_velocity surface used here.

Model
-----
A single rotational degree of freedom about the pivot axis:

    I * theta_ddot = commanded_torque - damping_coefficient * theta_dot

integrated with semi-implicit (symplectic) Euler, which is what
physics/pendulum_backend.py already uses for the same reason: it is
unconditionally stable for this class of lightly-damped oscillator at
reasonable time steps, unlike explicit Euler.

Gravity is deliberately NOT included as a restoring term here. Unlike the
pendulum bob (which physically hangs off a pivot and wants to swing back
to vertical), TankConfig's default pivot sits at the tank's own footprint
center at floor level -- a rocking cradle, not a hanging weight -- so
there is no first-order restoring torque to model at this fidelity level.
If a future TankConfig/PivotConfig combination puts the tank's center of
mass meaningfully off-axis, revisit this assumption rather than silently
keep ignoring it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.simulation_config import EnvironmentConfig, PivotConfig, TankConfig
from utilities.inertia_utils import parallel_axis_shift
from utilities.tank_geometry import tank_component_boxes
from utilities.inertia_utils import compute_composite_mass_properties


def tank_moment_of_inertia_about_pivot(
    tank_config: TankConfig, pivot_config: PivotConfig
) -> float:
    """Scalar rotational inertia of the tank about the pivot rotation axis.

    Reuses the tank's real composite mass properties
    (utilities/tank_geometry.py + utilities/inertia_utils.py -- the exact
    same numbers builders/tank_builder.py hands to
    ChBody.SetInertiaXX/SetInertiaXY) rather than approximating the tank
    as a point mass, then:

      1. Parallel-axis-shifts the tank's own centroidal inertia tensor to
         the pivot point (offset = tank's world-frame COM minus the
         pivot's world-frame position; both configs are already expressed
         in the same world frame -- see TankConfig.position's and
         PivotConfig.position's own docstrings).
      2. Projects that 3x3 tensor onto PivotConfig.axis via the standard
         quadratic form I_axis = axis_hat . I . axis_hat, so an
         off-Z pivot axis (should PivotConfig.axis ever change from its
         default) is still handled correctly instead of silently assuming
         Izz.

    Only valid at zero tilt (tank frame aligned with world frame), which
    matches this module's own scope: a Phase 8 surrogate, not a full
    rigid-body simulation where the inertia tensor itself would need to
    rotate with the body.
    """
    mass_properties = compute_composite_mass_properties(tank_component_boxes(tank_config))

    tank_world_com = np.array(tank_config.position) + np.array(mass_properties.center_of_mass)
    offset = tuple((tank_world_com - np.array(pivot_config.position)).tolist())

    inertia_about_pivot = parallel_axis_shift(
        mass_properties.inertia_tensor, mass_properties.mass, offset
    )

    axis = np.array(pivot_config.axis, dtype=float)
    axis_hat = axis / np.linalg.norm(axis)
    return float(axis_hat @ inertia_about_pivot @ axis_hat)


class TankDynamicsModel:
    """Single-DOF torque-driven surrogate for the tank's own rocking motion.

    See this module's docstring for the model and its scope. Kept as
    plain state (no pychrono, no gymnasium) so it is trivially unit
    testable and reusable outside the Gym env if needed (e.g. a future
    non-RL open-loop torque script).
    """

    def __init__(self, rotational_inertia: float, damping_coefficient: float, torque_limit: float) -> None:
        if rotational_inertia <= 0:
            raise ValueError("rotational_inertia must be positive.")
        if damping_coefficient < 0:
            raise ValueError("damping_coefficient must be non-negative.")
        if torque_limit <= 0:
            raise ValueError("torque_limit must be positive.")

        self._inertia = rotational_inertia
        self._damping = damping_coefficient
        self._torque_limit = torque_limit

        self._theta = 0.0
        self._theta_dot = 0.0

    def reset(self, seed: Optional[int] = None) -> None:
        self._theta = 0.0
        self._theta_dot = 0.0

    def step(self, dt: float, commanded_torque: float) -> None:
        """Advance the tank angle by dt seconds under a clipped commanded torque."""
        torque = float(np.clip(commanded_torque, -self._torque_limit, self._torque_limit))

        angular_accel = (torque - self._damping * self._theta_dot) / self._inertia

        # Semi-implicit Euler: update velocity first, then use the NEW
        # velocity to update position -- same integration scheme
        # physics/pendulum_backend.py::PendulumBackend.step() uses, for
        # the same stability reason (see this module's own docstring).
        self._theta_dot += angular_accel * dt
        self._theta += self._theta_dot * dt

    @property
    def angle(self) -> float:
        return self._theta

    @property
    def angular_velocity(self) -> float:
        return self._theta_dot

    @property
    def torque_limit(self) -> float:
        return self._torque_limit


def build_tank_dynamics_model(
    tank_config: TankConfig, pivot_config: PivotConfig, env_config: EnvironmentConfig
) -> TankDynamicsModel:
    """Construct a TankDynamicsModel sized to this project's tank/pivot/env configs."""
    inertia = tank_moment_of_inertia_about_pivot(tank_config, pivot_config)
    return TankDynamicsModel(
        rotational_inertia=inertia,
        damping_coefficient=env_config.pivot_damping_coefficient,
        torque_limit=env_config.max_motor_torque_nm,
    )
