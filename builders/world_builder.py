"""
Phase 4 -- PyChrono World Builder.

Responsible for exactly one thing: constructing an empty, correctly
configured PyChrono physics world (contact method, gravity, collision
system, solver). Nothing physical is added here -- the tank, motor, and
pivot are the job of later builders (tank_builder.py, motor_builder.py,
...), all of which receive the ChSystem this module produces rather than
creating their own. This keeps "how the world is configured" and "what's
in the world" as two separately testable concerns (Layer 1 vs Layer 2 in
the Module 1 architecture).
"""

import logging

import pychrono.core as chrono

from config.simulation_config import ContactMethod, WorldConfig

logger = logging.getLogger(__name__)


class WorldBuilder:
    """Builds and configures the top-level PyChrono simulation system."""

    def __init__(self, config: WorldConfig) -> None:
        self._config = config

    def build(self) -> chrono.ChSystem:
        """Create and configure a ChSystem per this builder's config.

        Returns:
            A ChSystemSMC or ChSystemNSC instance (per config.contact_method),
            gravity, collision system, and solver already configured, with
            zero bodies in it.
        """
        system = self._create_system()
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        self._configure_gravity(system)
        self._configure_solver(system)

        logger.info(
            "World built: contact_method=%s, gravity=%s, time_step=%s",
            self._config.contact_method.value,
            self._config.gravity,
            self._config.time_step,
        )
        return system

    def _create_system(self) -> chrono.ChSystem:
        if self._config.contact_method is ContactMethod.SMC:
            return chrono.ChSystemSMC()
        if self._config.contact_method is ContactMethod.NSC:
            return chrono.ChSystemNSC()
        raise ValueError(f"Unknown contact method: {self._config.contact_method}")  # pragma: no cover

    def _configure_gravity(self, system: chrono.ChSystem) -> None:
        gx, gy, gz = self._config.gravity if self._config.enable_gravity else (0.0, 0.0, 0.0)
        system.SetGravitationalAcceleration(chrono.ChVector3d(gx, gy, gz))

    def _configure_solver(self, system: chrono.ChSystem) -> None:
        solver = chrono.ChSolverBB()
        solver.SetMaxIterations(self._config.solver_max_iterations)
        system.SetSolver(solver)
