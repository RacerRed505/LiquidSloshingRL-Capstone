"""
Plain-data geometry mapping from the tank's existing box components
(utilities/tank_geometry.py) into the shape Chrono::FSI's rigid-body
coupling API will eventually need -- prepared ahead of having FSI
installed, per documentation/chrono_fsi_build_guide.md's open design
question #1 (the tank needs to become an FSI-coupled rigid body, since
the container in Chrono's own FSI demo is static and this project's tank
rocks continuously under a real motor joint).

Deliberately does NOT import or construct `chrono.ChBodyGeometry` /
`chrono.BoxShape` (the classes Chrono::FSI's `AddRigidBody()` actually
wants, per demo_FSI_ObjectDrop.py) -- confirmed directly, not assumed,
that neither class exists in this project's currently-installed
`pychrono.core` (only `ChContactMaterialData`, `ChBox`, `VNULL`, `QUNIT`
do). This module produces plain Python data in exactly the shape those
classes need instead: once a real FSI-enabled build exists, wrapping
each `FsiBoxGeometry` below into a real
`chrono.BoxShape(chrono.ChVector3d(*center), chrono.QUNIT,
chrono.ChBox(chrono.ChVector3d(*full_dimensions)))` call is a mechanical
one-line-per-component translation, not a redesign.

No rotation field: all 5 of the tank's box components are axis-aligned in
the tank's own local frame (confirmed by re-reading
builders/tank_builder.py's own `build()`, which uses
`chrono.ChQuaterniond(1, 0, 0, 0)` -- identity -- for every one of its 5
component frames, never anything else) -- so every eventual `BoxShape`
call would use identity rotation (`chrono.QUNIT`) too. Left out rather
than carried as an always-identical field.

PyChrono-free by design, same convention (and same reason) as
tank_geometry.py itself: this needs to be checkable/testable on any
machine, not just one with the real conda-only bindings, since it
produces plain data, not chrono objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from config.simulation_config import TankConfig
from utilities.tank_geometry import tank_component_boxes


@dataclass(frozen=True)
class FsiBoxGeometry:
    """One box component's geometry, in exactly the shape
    `chrono.BoxShape(position, rotation, chrono.ChBox(size))` will want
    once Chrono::FSI is actually installed -- see this module's own
    docstring for why it isn't wrapped in that real class yet.
    """

    center: Tuple[float, float, float]  # tank-local frame, matches TankBuilder's own convention
    full_dimensions: Tuple[float, float, float]  # (x, y, z) FULL lengths, NOT half-extents --
    # matches chrono.ChBox's own constructor convention (confirmed against
    # demo_FSI_ObjectDrop.py's own "bottom_offset = size.z / 2" usage,
    # which only makes sense if size.z is the full height) and matches
    # BoxComponent.dims's existing meaning unchanged.
    is_base: bool  # True for the single base slab, False for the 4 walls --
    # mirrors tank_builder.py's own index==0 base/wall distinction.


def tank_box_geometry_for_fsi(tank_config: TankConfig) -> List[FsiBoxGeometry]:
    """The tank's 5 box components, translated into FSI-ready plain data.

    Reuses utilities/tank_geometry.py::tank_component_boxes() -- the SAME
    single source of truth builders/tank_builder.py's real ChBody already
    uses -- rather than re-deriving the tank's geometry a third time.
    """
    boxes = tank_component_boxes(tank_config)
    return [
        FsiBoxGeometry(center=box.center, full_dimensions=box.dims, is_base=(index == 0))
        for index, box in enumerate(boxes)
    ]
