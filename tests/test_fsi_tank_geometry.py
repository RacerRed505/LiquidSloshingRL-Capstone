"""
Unit tests for utilities/fsi_tank_geometry.py.

Pure Python, no pychrono import at all -- runs anywhere, including a
sandbox without the real (conda-only) PyChrono bindings, same as
tests/test_inertia_utils.py / tests/test_tank_dynamics.py. This module
exists to prepare Phase 13-14 (Chrono::FSI) data ahead of having FSI
installed -- see utilities/fsi_tank_geometry.py's own docstring for what
it does and deliberately does not do yet.

Run:
    pytest tests/test_fsi_tank_geometry.py -v
"""

from config.simulation_config import TankConfig
from utilities.fsi_tank_geometry import FsiBoxGeometry, tank_box_geometry_for_fsi
from utilities.tank_geometry import tank_component_boxes


def test_returns_exactly_five_components():
    geometry = tank_box_geometry_for_fsi(TankConfig())
    assert len(geometry) == 5
    assert all(isinstance(g, FsiBoxGeometry) for g in geometry)


def test_exactly_one_component_is_the_base():
    geometry = tank_box_geometry_for_fsi(TankConfig())
    base_flags = [g.is_base for g in geometry]
    assert base_flags.count(True) == 1
    assert base_flags[0] is True  # index 0, matching tank_builder.py's own convention
    assert all(flag is False for flag in base_flags[1:])


def test_center_and_dimensions_match_tank_component_boxes_exactly():
    """This module must not transform the numbers, only relabel them --
    tank_component_boxes() stays the single source of truth for the
    tank's actual geometry."""
    config = TankConfig()
    boxes = tank_component_boxes(config)
    geometry = tank_box_geometry_for_fsi(config)
    assert len(boxes) == len(geometry)
    for box, g in zip(boxes, geometry):
        assert g.center == box.center
        assert g.full_dimensions == box.dims


def test_dimensions_are_full_lengths_not_half_extents():
    """Sanity check against a concrete, known number: the base slab's
    full length must equal TankConfig.length exactly (not half of it) --
    this is what makes these dimensions safe to pass straight to
    chrono.ChBox() once available, which also expects full lengths."""
    config = TankConfig(length=0.5, width=0.3, height=0.25, wall_thickness=0.01)
    geometry = tank_box_geometry_for_fsi(config)
    base = next(g for g in geometry if g.is_base)
    assert base.full_dimensions[0] == config.length
    assert base.full_dimensions[2] == config.width


def test_reflects_a_nondefault_tank_config():
    config = TankConfig(length=0.6, width=0.35, height=0.4, wall_thickness=0.008)
    geometry = tank_box_geometry_for_fsi(config)
    # Same structural invariants should hold for any valid TankConfig, not
    # just the default -- catches a hidden dependency on default values.
    assert len(geometry) == 5
    assert sum(g.is_base for g in geometry) == 1
