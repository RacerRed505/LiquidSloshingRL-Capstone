"""
Phase 5 -- Procedural Tank Generation.

Builds the sloshing tank as a single rigid ChBody composed of five box
primitives (one base + four walls) -- no external CAD file, no imported
mesh. Mass, center of mass, and the full 3x3 inertia tensor are derived
analytically from the tank's own geometry (utilities/inertia_utils.py)
rather than left to any engine default, so the exact numbers going into
SetMass/SetInertiaXX/SetInertiaXY are independently unit tested (see
tests/test_inertia_utils.py) before they ever touch PyChrono.
"""

import logging
from typing import List

import pychrono.core as chrono

from config.simulation_config import TankConfig
from utilities.inertia_utils import BoxComponent, MassProperties, compute_composite_mass_properties
from utilities.tank_geometry import tank_component_boxes

logger = logging.getLogger(__name__)

_WALL_COLOR = (0.25, 0.55, 0.95)
_BASE_COLOR = (0.35, 0.35, 0.35)


class TankBuilder:
    """Builds the procedural open-top sloshing tank."""

    def __init__(self, config: TankConfig) -> None:
        self._config = config

    def compute_mass_properties(self) -> MassProperties:
        """Pure analytical mass/COM/inertia computation -- no PyChrono calls.

        Kept separate from build() so it can be unit tested, and reused by
        later builders (e.g. motor_builder.py in Phase 6, which needs the
        tank's mass to size the drive motor) without requiring a live
        ChSystem.
        """
        return compute_composite_mass_properties(self._component_boxes())

    def build(self, system: chrono.ChSystem) -> chrono.ChBody:
        """Construct the tank body, attach shapes, and add it to `system`."""
        props = self.compute_mass_properties()
        body = chrono.ChBody()

        world_com = chrono.ChVector3d(
            self._config.position[0] + props.center_of_mass[0],
            self._config.position[1] + props.center_of_mass[1],
            self._config.position[2] + props.center_of_mass[2],
        )
        body.SetPos(world_com)
        body.SetMass(props.mass)
        body.SetInertiaXX(chrono.ChVector3d(
            props.inertia_tensor[0, 0],
            props.inertia_tensor[1, 1],
            props.inertia_tensor[2, 2],
        ))
        body.SetInertiaXY(chrono.ChVector3d(
            props.inertia_tensor[0, 1],
            props.inertia_tensor[0, 2],
            props.inertia_tensor[1, 2],
        ))
        body.SetFixed(self._config.fixed)
        body.EnableCollision(True)

        material = self._build_contact_material()
        components = self._component_boxes()
        for index, component in enumerate(components):
            # Shapes are positioned relative to the body's own origin, which
            # we've placed at the composite COM above -- NOT the
            # footprint-local origin the components were defined in -- so
            # every shape offset must be re-based onto the COM here.
            local_center = tuple(
                component.center[axis] - props.center_of_mass[axis] for axis in range(3)
            )
            frame = chrono.ChFramed(chrono.ChVector3d(*local_center), chrono.ChQuaterniond(1, 0, 0, 0))

            collision_shape = chrono.ChCollisionShapeBox(material, *component.dims)
            body.AddCollisionShape(collision_shape, frame)

            visual_shape = chrono.ChVisualShapeBox(*component.dims)
            color = _BASE_COLOR if index == 0 else _WALL_COLOR
            visual_shape.SetColor(chrono.ChColor(*color))
            body.AddVisualShape(visual_shape, frame)

        system.Add(body)
        logger.info(
            "Tank built: mass=%.4f kg, COM(local)=%s, fixed=%s",
            props.mass, props.center_of_mass, self._config.fixed,
        )
        return body

    def _build_contact_material(self) -> "chrono.ChContactMaterialSMC":
        material = chrono.ChContactMaterialSMC()
        material.SetFriction(self._config.friction)
        material.SetRestitution(self._config.restitution)
        material.SetYoungModulus(self._config.contact_young_modulus)
        material.SetPoissonRatio(self._config.contact_poisson_ratio)
        return material

    def _component_boxes(self) -> List[BoxComponent]:
        """The tank's 5 box primitives, in the tank's local frame.

        Delegates to utilities/tank_geometry.py::tank_component_boxes so
        this geometry has one source of truth -- Phase 8's
        environment/tank_dynamics.py needs the same box layout (to derive
        the tank's rotational inertia about the pivot) without importing
        pychrono, so the box math itself now lives in a pychrono-free
        module and this method is a thin wrapper kept for backwards
        compatibility with the rest of this class.
        """
        return tank_component_boxes(self._config)
