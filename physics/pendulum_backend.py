"""
Equivalent-pendulum fluid approximation.

Models the sloshing liquid as a point mass on a rigid, damped pendulum
whose pivot is fixed to the tank. This is the Phase 7 surrogate used
to train the initial PPO policy before Chrono::FSI is introduced in
Phase 13. See Phase 14 (Backend Replacement) for how this gets swapped
for SPHBackend without touching the environment layer.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from physics.backend_interface import FluidBackend, FluidState


class PendulumBackend(FluidBackend):
    """Single equivalent pendulum representing the fluid's dominant slosh mode."""

    def __init__(
        self,
        pendulum_length: float,
        equivalent_mass: float,
        damping_coefficient: float,
        gravity: float = 9.81,
    ) -> None:
        """
        Args:
            pendulum_length: Effective pendulum length in meters, tuned to
                match the fluid's first sloshing frequency.
            equivalent_mass: Point mass in kg representing the moving fluid.
            damping_coefficient: Linear damping on angular velocity (1/s).
            gravity: Gravitational acceleration in m/s^2.
        """
        self._length = pendulum_length
        self._mass = equivalent_mass
        self._damping = damping_coefficient
        self._gravity = gravity

        self._theta = 0.0        # pendulum angle relative to tank frame (rad)
        self._theta_dot = 0.0    # angular velocity (rad/s)

    def reset(self, seed: Optional[int] = None) -> None:
        self._theta = 0.0
        self._theta_dot = 0.0

    def step(self, dt: float, tank_motion: Tuple[float, float]) -> None:
        _, tank_angular_velocity = tank_motion

        # NOTE: the coupling term below is a first-pass approximation --
        # it treats tank angular velocity as a direct forcing term on the
        # pendulum. This gets refined with a proper non-inertial-frame
        # derivation in the Phase 7 module; flagged here rather than
        # silently presented as final physics.
        angular_accel = (
            -(self._gravity / self._length) * math.sin(self._theta)
            - self._damping * self._theta_dot
            - tank_angular_velocity
        )

        self._theta_dot += angular_accel * dt
        self._theta += self._theta_dot * dt

    def get_state(self) -> FluidState:
        x = self._length * math.sin(self._theta)
        y = -self._length * math.cos(self._theta)
        vx = self._length * self._theta_dot * math.cos(self._theta)
        vy = self._length * self._theta_dot * math.sin(self._theta)
        slosh_energy = 0.5 * self._mass * (vx**2 + vy**2)
        # Gravitational potential energy of the convective (Housner) mass
        # relative to its hanging-at-rest position: m*g*L*(1 - cos theta).
        # Kinetic + this is the pendulum's full mechanical energy, which
        # the kinetic-only slosh_energy above is not (at a swing's turning
        # points slosh_energy is ~0 while the mode still holds energy).
        slosh_potential_energy = (
            self._mass * self._gravity * self._length * (1.0 - math.cos(self._theta))
        )

        return FluidState(
            com_position=(x, y, 0.0),
            com_velocity=(vx, vy, 0.0),
            slosh_energy=slosh_energy,
            slosh_potential_energy=slosh_potential_energy,
        )

    def apply_disturbance(self, impulse: Tuple[float, float, float]) -> None:
        # Project the planar impulse onto the pendulum's tangential direction.
        impulse_theta = (
            impulse[0] * math.cos(self._theta) + impulse[1] * math.sin(self._theta)
        )
        self._theta_dot += impulse_theta / (self._mass * self._length)

    @property
    def name(self) -> str:
        return "pendulum"
