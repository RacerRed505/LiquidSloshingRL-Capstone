"""
Abstract interface for interchangeable fluid-behavior backends.

Both the low-fidelity pendulum approximation (Phase 7) and the
high-fidelity SPH / Chrono::FSI simulation (Phase 13) implement this
same interface. The Gymnasium environment (Layer 3) is written
against FluidBackend only -- it never imports PendulumBackend or
SPHBackend directly. This is what makes the Phase 14 "backend
replacement" a one-line composition swap instead of an environment
rewrite, and what makes the claim "observation space, reward, and
action space are unchanged" verifiable rather than asserted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FluidState:
    """Backend-agnostic snapshot of fluid behavior at one timestep."""

    com_position: Tuple[float, float, float]   # center of mass, tank frame
    com_velocity: Tuple[float, float, float]
    # KINETIC slosh energy only (0.5*m*|v_com|^2). This is what the reward
    # and observation use, and it is deliberately left unchanged so every
    # already-trained policy and prior result stays valid. It is NOT the
    # full mechanical energy of the slosh mode -- see slosh_total_energy.
    slosh_energy: float
    # Gravitational potential energy of the slosh mode relative to its
    # at-rest equilibrium, when the backend can measure it reliably.
    # None means "not available for this backend" (see SPHBackend's
    # get_state() for why), never "zero".
    slosh_potential_energy: Optional[float] = None

    @property
    def slosh_total_energy(self) -> Optional[float]:
        """Kinetic + potential slosh energy, or None if the backend does
        not report a potential term."""
        if self.slosh_potential_energy is None:
            return None
        return self.slosh_energy + self.slosh_potential_energy


class FluidBackend(ABC):
    """
    Common contract every fluid model must satisfy.

    Anything that represents "how the liquid inside the tank behaves" --
    whether that's a point-mass pendulum or several thousand SPH
    particles -- must implement this and nothing more. The environment
    layer only ever calls these methods; it has no idea which concrete
    backend it's holding.
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reinitialize the fluid to a nominal at-rest configuration."""
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float, tank_motion: Tuple[float, float]) -> None:
        """
        Advance the fluid model by dt seconds.

        Args:
            dt: Simulation timestep in seconds.
            tank_motion: (tank_angle, tank_angular_velocity) -- the
                forcing input the tank imposes on the fluid this step.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> FluidState:
        """Return the current backend-agnostic fluid state."""
        raise NotImplementedError

    @abstractmethod
    def apply_disturbance(self, impulse: Tuple[float, float, float]) -> None:
        """Inject an external disturbance (for domain randomization / robustness tests)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier, e.g. 'pendulum' or 'sph'."""
        raise NotImplementedError
