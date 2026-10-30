"""
Pure-geometry decomposition of TankConfig into its five box primitives.

This was originally a private method on builders/tank_builder.py
(`_component_boxes`), but Phase 8 (environment/tank_dynamics.py) also
needs the tank's mass properties -- specifically its rotational inertia
about the pivot -- to build a torque-driven surrogate tank model, and it
needs them WITHOUT importing pychrono at all (same rationale as
utilities/housner_utils.py and utilities/inertia_utils.py: the
environment layer must stay importable and unit-testable in any sandbox
that lacks the real conda-only PyChrono bindings).

Moving this here, rather than duplicating the box layout a second time,
keeps a single source of truth for "what are the tank's 5 boxes" --
builders/tank_builder.py now calls tank_component_boxes() instead of
defining its own copy. tests/test_inertia_utils.py's own hand-rolled
5-box tank fixture is a deliberate independent duplicate for
cross-checking purposes and is intentionally left alone.
"""

from typing import List

from config.simulation_config import TankConfig
from utilities.inertia_utils import BoxComponent


def tank_component_boxes(config: TankConfig) -> List[BoxComponent]:
    """The tank's 5 box primitives, in the tank's local frame.

    Local frame: origin at the footprint center, y=0 at interior floor
    level. Component index 0 is always the base (builders/tank_builder.py
    uses this to color it differently from the four walls).

    Symmetry note: walls are defined as two identical +/-z pairs and two
    identical +/-x pairs. That symmetry is what makes every
    product-of-inertia term (Ixy, Ixz, Iyz) cancel to ~0 in the combined
    tensor -- verified directly in
    tests/test_inertia_utils.py::test_open_top_tank_com_and_symmetry.
    Interior-cavity volume is treated as L x h x W minus the thin walls,
    ignoring the small corner double-counting where walls meet -- a
    negligible simplification at these wall thicknesses, noted here
    rather than left implicit.
    """
    c = config
    L, W, h, t = c.length, c.width, c.height, c.wall_thickness

    def box_mass(a: float, b: float, depth: float) -> float:
        return c.density * a * b * depth

    return [
        BoxComponent(mass=box_mass(L, t, W), center=(0.0, -t / 2, 0.0), dims=(L, t, W)),
        BoxComponent(mass=box_mass(L, h, t), center=(0.0, h / 2, W / 2 - t / 2), dims=(L, h, t)),
        BoxComponent(mass=box_mass(L, h, t), center=(0.0, h / 2, -(W / 2 - t / 2)), dims=(L, h, t)),
        BoxComponent(mass=box_mass(t, h, W), center=(L / 2 - t / 2, h / 2, 0.0), dims=(t, h, W)),
        BoxComponent(mass=box_mass(t, h, W), center=(-(L / 2 - t / 2), h / 2, 0.0), dims=(t, h, W)),
    ]
