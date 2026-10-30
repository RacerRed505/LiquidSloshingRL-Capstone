"""
Phase 6a -- Pivot Support Builder.

Builds a small, fixed pedestal body anchored to ground at
PivotConfig.position, oriented so its own local Z axis points along
PivotConfig.axis. This is the physical mounting point
builders/motor_builder.py's revolute motor joint attaches the tank to,
rather than joining the tank directly to an implicit world frame.

Like tank_builder.py, this is a single procedural ChBody (one box
primitive), reusing utilities/inertia_utils.py's own box-inertia formula
rather than re-deriving it, with no CAD import involved.
"""

import logging

import pychrono.core as chrono

from config.simulation_config import PivotConfig
from utilities.inertia_utils import box_inertia_about_centroid
from utilities.quaternion_utils import quaternion_aligning_z_to

logger = logging.getLogger(__name__)

_SUPPORT_COLOR = (0.6, 0.6, 0.6)


class PivotBuilder:
    """Builds the fixed pedestal that the pivot/motor joint attaches to."""

    def __init__(self, config: PivotConfig) -> None:
        self._config = config

    def joint_frame(self) -> chrono.ChFramed:
        """The world-frame ChFramed at the pivot, local Z aligned to config.axis.

        Shared by builders/motor_builder.py so the motor joint is built at
        exactly the same frame this pedestal was actually placed at --
        computed once here rather than re-derived in two places, and
        callable without a live ChSystem (no build() call required first).
        """
        w, x, y, z = quaternion_aligning_z_to(self._config.axis)
        return chrono.ChFramed(
            chrono.ChVector3d(*self._config.position),
            chrono.ChQuaterniond(w, x, y, z),
        )

    def build(self, system: chrono.ChSystem) -> chrono.ChBody:
        """Construct the fixed pedestal body and add it to `system`."""
        c = self._config
        volume = c.support_size[0] * c.support_size[1] * c.support_size[2]
        mass = c.support_density * volume
        inertia = box_inertia_about_centroid(mass, c.support_size)

        body = chrono.ChBody()
        body.SetPos(chrono.ChVector3d(*c.position))
        body.SetMass(mass)
        body.SetInertiaXX(chrono.ChVector3d(inertia[0, 0], inertia[1, 1], inertia[2, 2]))
        body.SetFixed(True)  # bolted to the world -- never itself moves
        body.EnableCollision(False)  # visual mounting point only, no contact role

        visual_shape = chrono.ChVisualShapeBox(*c.support_size)
        visual_shape.SetColor(chrono.ChColor(*_SUPPORT_COLOR))
        body.AddVisualShape(visual_shape, chrono.ChFramed())

        system.Add(body)
        logger.info(
            "Pivot support built: position=%s axis=%s mass=%.4f kg",
            c.position, c.axis, mass,
        )
        return body
