"""
Phase 7 -- Equivalent-Pendulum Fluid Backend Construction.

Bridges the tank's own geometry (TankConfig, already used by
builders/tank_builder.py) and fill level (FluidConfig) into the concrete
parameters physics/pendulum_backend.py::PendulumBackend needs
(pendulum_length, equivalent_mass, damping_coefficient), via the closed-form
Housner (1963) / ACI 350.3-06 relations in utilities/housner_utils.py.

This builder makes no ChSystem or ChBody calls -- unlike WorldBuilder and
TankBuilder, PendulumBackend is a standalone analytical model (see its own
docstring), so there is nothing here for PyChrono to build. That's also
why this file, like utilities/housner_utils.py, runs and is unit-testable
without the real (conda-only) PyChrono bindings installed.

Not yet wired into main.py: main.py's simulation loop doesn't drive a
tank_motion signal yet, since that's produced by the pivot/motor joint
(Phase 6, builders/motor_builder.py + builders/pivot_builder.py), which are
still empty stubs. FluidBuilder itself has no dependency on Phase 6 being
done -- it can be built, logged, and unit tested against a fixed
tank_motion right now -- but a live demo showing the pendulum actually
respond to tank movement needs Phase 6 finished first.
"""

import logging

from config.simulation_config import FluidConfig, TankConfig, WorldConfig
from physics.pendulum_backend import PendulumBackend
from utilities.housner_utils import HousnerParameters, compute_housner_parameters

logger = logging.getLogger(__name__)


class FluidBuilder:
    """Builds the Phase 7 equivalent-pendulum FluidBackend from tank + fluid config."""

    def __init__(
        self,
        tank_config: TankConfig,
        fluid_config: FluidConfig,
        world_config: WorldConfig,
    ) -> None:
        if fluid_config.fill_height >= tank_config.height:
            raise ValueError(
                f"fill_height ({fluid_config.fill_height} m) must be less than "
                f"TankConfig.height ({tank_config.height} m) -- the tank would "
                "overflow before the simulation even starts."
            )
        self._tank_config = tank_config
        self._fluid_config = fluid_config
        # Gravity comes from WorldConfig rather than a local constant, per
        # the "no magic numbers in builders/" rule -- this also keeps
        # FluidBuilder consistent if WorldConfig.gravity is ever changed
        # (e.g. a reduced-gravity training variant).
        self._gravity = abs(world_config.gravity[1])

    def compute_backend_parameters(self) -> HousnerParameters:
        """Pure analytical Housner computation -- no PendulumBackend instantiation.

        Kept separate from build() so it can be unit tested on its own
        (tests/test_housner_utils.py exercises the underlying function
        directly; this method is a thin, project-specific wrapper around
        it, reading tank_length from TankConfig.length -- the axis the
        pivot/motor will rotate the tank about in Phase 6).
        """
        return compute_housner_parameters(
            tank_length=self._tank_config.length,
            tank_breadth=self._tank_config.width,
            fill_height=self._fluid_config.fill_height,
            fluid_density=self._fluid_config.density,
            gravity=self._gravity,
        )

    def build(self) -> PendulumBackend:
        """Construct the PendulumBackend sized to this tank's geometry and fill level."""
        params = self.compute_backend_parameters()

        # 2 * zeta * omega is the standard linear-damping-coefficient form
        # for a damping ratio zeta on an oscillator of natural frequency
        # omega -- consistent with the "-self._damping * theta_dot" term
        # PendulumBackend.step() already implements.
        damping_coefficient = 2.0 * self._fluid_config.damping_ratio * params.natural_frequency_rad_s

        backend = PendulumBackend(
            pendulum_length=params.pendulum_length,
            equivalent_mass=params.convective_mass,
            damping_coefficient=damping_coefficient,
            gravity=self._gravity,
        )

        logger.info(
            "Fluid backend built: model=%s m_total=%.4f kg m_i=%.4f kg m_c=%.4f kg "
            "h_i=%.4f m h_c=%.4f m L_p=%.4f m f=%.4f Hz damping_ratio=%.4f "
            "modeled_fraction=%.3f",
            backend.name, params.total_liquid_mass, params.impulsive_mass,
            params.convective_mass, params.impulsive_height,
            params.convective_pivot_height, params.pendulum_length,
            params.natural_frequency_hz, self._fluid_config.damping_ratio,
            params.modeled_mass_fraction,
        )
        return backend
