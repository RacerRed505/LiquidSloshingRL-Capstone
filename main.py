"""
Phase 4-7 entry point: build the world, procedural tank, pivot/motor, and
equivalent-pendulum fluid backend, then render a live coupled demo.

Run:
    python main.py

Expected result (the Phase 6 checkpoint, with Phase 7 now wired in too):
an Irrlicht window opens showing the open-top tank rocking back and forth
under an open-loop sinusoidal motor (MotorConfig), about a fixed pivot
support -- and a small orange marker sphere inside the tank tracing the
fluid model's response to that motion (PendulumBackend), driven every
step by the REAL angle/angular-velocity read straight off the PyChrono
motor joint -- not a scripted or duplicated value.

The marker is a pure visualization aid: it has zero mass, no collision,
and does not feed back into the rigid-body solve. The actual physics
being demonstrated is entirely PendulumBackend's own ODE integration
(physics/pendulum_backend.py) -- the marker just makes that otherwise
invisible analytical state visible in the same window as the real
Chrono tank.

Requires the real PyChrono bindings (conda-forge), which is a compiled
package and NOT installable via pip:
    conda install -c conda-forge pychrono

Optional `--fluid-viz` flag: replaces the single marker dot with a CPU-only
deforming height-field surface standing in for the liquid's free surface --
see _RippleSurface's and _surface_tilt_rad()'s own docstrings for exactly
what it does and does not represent physically. This is a purely cosmetic
rendering option; it changes nothing about the physics loop below
(PendulumBackend, environment/, training/, evaluation/ are all completely
untouched), and the default (no flag) marker-dot demo is untouched.

Read this before assuming the deforming surface shows real wave physics:
it does not. `physics/pendulum_backend.py` has no spatial wave data at all
-- it only ever produces a single point (the equivalent pendulum's bob).
The height-field's BASE tilt is the same physically-motivated (but still
approximate) proxy derived from that one point, same as before. Everything
ELSE about the height-field -- the ripple pattern, its wave speed, damping,
and how strongly it responds to the tank's motion -- is a separate,
illustrative, purely decorative model built for this demo, one-way driven
by the tank's angle/angular velocity (reads them, never feeds back into the
real physics). `slosh_energy` in the console readout below always comes
from the real PendulumBackend's own state, never from this visual model.

Performance note (measured, not assumed): the actual demo loop runs at
~2100 Hz on this machine (physics + fluid-backend stepping + rendering
combined, no surface viz) -- confirmed with a real stopwatch, not
back-calculated from simulated-vs-wall-clock time. Writing all 256 grid
vertices into the mesh every single iteration (~0.6ms, extrapolated from a
measured 0.33ms/144-vertex figure) would be comparable to or larger than
that entire per-iteration budget on its own. So the expensive part (pushing
the height field into the actual PyChrono mesh) is throttled to every
_RIPPLE_MESH_SYNC_EVERY_N_STEPS-th iteration; the cheap part (evolving the
wave simulation itself, pure numpy) still runs every iteration at the full,
numerically-correct timestep. See _RippleSurface's own docstring for the
exact numbers and reasoning.

Optional `--fluid-viz-resonance` flag: same surface-slab rendering as
--fluid-viz, PLUS overrides the motor's drive frequency (for THIS demo
process only) to the pendulum's own natural frequency, so the sloshing
response -- and therefore the visible tilt -- is much more pronounced
than MotorConfig's real default (0.4 Hz, chosen for the RL training
disturbance case, which is deliberately off-resonance). This does NOT
change MotorConfig's actual default, nor anything training/evaluation/
tests use -- it is a demo-only convenience local to this script.

Optional `--fluid-viz-interactive` flag: same surface-slab rendering as
--fluid-viz, PLUS lets you retune the motor's max inclination (amplitude)
and swing speed (frequency) live while the simulation runs, via the
CONSOLE (not the graphics window -- see below for why). Starts in
AUTOMATIC mode, byte-for-byte the same scripted sine behavior as today
(amplitude/frequency taken from MotorConfig's real default, unchanged).
Press 'm' (console must have keyboard focus) to toggle into MANUAL mode:
  '['  decrease max inclination      ']'  increase max inclination
  ','  decrease swing speed          '.'  increase swing speed
Press 'm' again to return to AUTOMATIC. The console readout (throttled,
same cadence as the existing slosh-state log line) adds the current mode
and both control values whenever this flag is active.

Why the console, not on-screen sliders/text: checked directly against
this project's installed PyChrono/Irrlicht build (not assumed) --
IGUIEnvironment.addScrollBar()/addStaticText() both exist and don't
error, but return raw untyped SwigPyObject pointers with no usable
methods (no setText/getPos/setMax -- PyChrono's bindings never actually
wrapped IGUIStaticText/IGUIScrollBar themselves). Subclassing
IEventReceiver for custom in-window key handling also isn't possible
here: unlike chrono.ChFunction (director-enabled), IEventReceiver raises
"No constructor defined - class is abstract" the moment Python tries to
subclass it. Console-based keyboard polling (the stdlib `msvcrt` module,
Windows-only -- consistent with this project's existing Windows-only
scope) is the realistic fallback given what's actually available.

Implementation note: MANUAL-mode value changes (and AUTOMATIC/MANUAL
toggling itself) never swap the underlying ChFunction object -- see
builders/motor_builder.py::MotorBuilder.make_live_angle_function()'s own
docstring for why, and _LiveSineAngleFunction's docstring for the one
resulting visible quirk (a small instantaneous angle jump exactly when a
value actually changes, not merely when toggling mode).
"""

import argparse
import logging
import math
import msvcrt

import numpy as np
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr

from builders.fluid_builder import FluidBuilder
from builders.motor_builder import MotorBuilder
from builders.pivot_builder import PivotBuilder
from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import (
    FluidConfig,
    MotorConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MARKER_RADIUS = 0.012  # m, purely visual
_MARKER_COLOR = (0.95, 0.35, 0.15)
_LOG_EVERY_N_STEPS = 200  # throttle slosh-state logging to keep the console readable

# --fluid-viz cosmetic constants (rendering-only, no physics meaning). The
# material constants below are static for the whole run -- they never change
# with slosh_energy, tilt, or anything else; only the surface's shape
# (tilt + ripple, see _RippleSurface and _surface_tilt_rad()) reflects any
# simulation state.
_SURFACE_DIFFUSE_COLOR = (0.15, 0.45, 0.55)  # teal-blue "water" base color
_SURFACE_OPACITY = 0.55  # translucent, not fully see-through
_SURFACE_SPECULAR_COLOR = (0.9, 0.95, 1.0)  # near-white highlight, water-like sheen
_SURFACE_SPECULAR_EXPONENT = 0.7  # broad, soft highlight rather than a tight glint
_SURFACE_MAX_TILT_RAD = math.radians(15.0)  # visual cap on how far the base tilt shears

# --fluid-viz deforming-surface constants. ALL of these (grid size, damping,
# excitation gain) are freely chosen for visual effect, not derived from any
# real wave physics -- see _RippleSurface's own docstring.
_RIPPLE_GRID_N = 16  # vertices per side (256 total, 450 triangles) -- see
# _RippleSurface's docstring for the measured performance budget this fits.
_RIPPLE_MESH_SYNC_EVERY_N_STEPS = 4  # throttle for the expensive per-vertex
# PyChrono mesh write-back ONLY -- the wave simulation itself still steps
# every physics iteration (see _RippleSurface.step_wave()'s docstring).
_RIPPLE_DAMPING = 0.999  # per real physics step (1ms) -- decays a disturbance
# to ~13% amplitude after ~2s of simulated time (0.999**2000 ~= 0.135).
_RIPPLE_EXCITATION_GAIN = 0.002  # m per (rad/s) of tank angular velocity,
# injected at the two length-wise edges each physics step -- a first-guess
# cosmetic value, not derived from anything; may need visual re-tuning.

# --fluid-viz-interactive cosmetic constants (demo-only motor retuning, no
# effect on MotorConfig's real default or anything training/evaluation uses).
_AMPLITUDE_STEP_RAD = math.radians(1.0)
_AMPLITUDE_MIN_RAD = math.radians(1.0)
_AMPLITUDE_MAX_RAD = math.radians(60.0)
_FREQUENCY_STEP_HZ = 0.05
_FREQUENCY_MIN_HZ = 0.05
_FREQUENCY_MAX_HZ = 5.0
_MODE_AUTOMATIC = "AUTOMATIC"
_MODE_MANUAL = "MANUAL"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fluid-viz",
        action="store_true",
        help=(
            "Render the sloshing fluid as a tilting surface slab instead of the "
            "default single marker dot. Purely cosmetic -- see this module's own "
            "docstring."
        ),
    )
    parser.add_argument(
        "--fluid-viz-resonance",
        action="store_true",
        help=(
            "Like --fluid-viz, but also overrides the motor's drive frequency to "
            "the pendulum's own natural frequency for THIS demo run only, so the "
            "tilt is visually obvious instead of the default off-resonance wobble. "
            "Does not change MotorConfig's real default or anything training/"
            "evaluation/tests use -- see this module's own docstring."
        ),
    )
    parser.add_argument(
        "--fluid-viz-interactive",
        action="store_true",
        help=(
            "Like --fluid-viz, but also lets you retune the motor's max "
            "inclination and swing speed live via the CONSOLE (not the graphics "
            "window) while the simulation runs. Starts in AUTOMATIC mode "
            "(identical to the default scripted motor); press 'm' (console must "
            "have keyboard focus) to toggle into MANUAL mode, where '['/']' "
            "adjust amplitude and ','/'.' adjust frequency. See this module's "
            "own docstring for why the console, not on-screen controls."
        ),
    )
    return parser.parse_args()


def _build_marker_body(system: chrono.ChSystem) -> chrono.ChBody:
    """A massless, collision-free sphere: visualizes PendulumBackend's bob only."""
    marker = chrono.ChBody()
    marker.SetFixed(True)  # kinematic -- we set its position by hand every frame
    marker.EnableCollision(False)
    visual_shape = chrono.ChVisualShapeSphere(_MARKER_RADIUS)
    visual_shape.SetColor(chrono.ChColor(*_MARKER_COLOR))
    marker.AddVisualShape(visual_shape, chrono.ChFramed())
    system.Add(marker)
    return marker


class _RippleSurface:
    """A massless, collision-free CPU-only deforming height-field surface
    standing in for the liquid's free surface -- replaces the earlier flat,
    rigidly-tilting slab for all three --fluid-viz* flags.

    What's real vs. decorative, explicitly (read this before trusting any of
    it): `physics/pendulum_backend.py` produces exactly one point (the
    equivalent pendulum's bob) -- it has no spatial wave data to draw a real
    height-field from. This class's BASE tilt (see sync_mesh()) reuses the
    same physically-motivated proxy _surface_tilt_rad() already derives from
    that one point. Everything else here -- the ripple pattern, its damping,
    and how strongly it reacts to the tank's motion -- is a separate,
    illustrative, one-way-coupled model built purely to look plausible: it
    reads tank_angular_velocity every step (same one-way-coupling limitation
    this project already has elsewhere) and never feeds anything back into
    PendulumBackend, environment/, training/, or evaluation/. `slosh_energy`
    in main()'s console readout always comes from PendulumBackend's own
    state -- this class never touches or contributes to that number.

    Material: a real `ChVisualMaterial` (teal-blue diffuse color,
    translucent opacity, soft specular highlight -- see
    _SURFACE_DIFFUSE_COLOR/_SURFACE_OPACITY/_SURFACE_SPECULAR_COLOR/
    _SURFACE_SPECULAR_EXPONENT above), confirmed as a genuinely working
    API before use (its setters are real methods, verified in a full
    constructed scene) -- unlike the dead GUI-widget stubs investigated
    for --fluid-viz-interactive. **This material is 100% static for the
    entire run**: it never changes with slosh_energy, tilt, or anything
    else -- only vertex heights (the tilt shear + ripple, see sync_mesh()
    below) reflect any simulation state. No animated TEXTURE was
    attempted (as opposed to the real per-vertex mesh deformation this
    class already does): `ChVisualMaterial` exposes no UV-offset or
    scrolling-texture mechanism in this build, only a static
    SetKdTexture(filename) and a fixed SetTextureScale tiling factor, so
    that specific effect was deliberately not built rather than faked.

    Verified against the actual installed PyChrono API before building this
    (same discipline as the GUI/keyboard investigation for
    --fluid-viz-interactive, with the opposite result this time):
    `ChTriangleMeshConnected.GetCoordsVertices()` returns a real, working
    SWIG STL-vector wrapper whose mutations write through to the mesh
    (confirmed: mutate through one reference, re-fetch a fresh
    GetCoordsVertices() call, see the change persisted) -- unlike the GUI
    widgets, which were unusable opaque stubs. `ChVisualShapeTriangleMesh
    .IsMutable()` is True by default, and a real render loop with per-frame
    vertex mutation ran with zero exceptions at good frame times.

    Performance budget (measured with a real stopwatch, not assumed): this
    demo's actual loop -- physics + fluid-backend stepping + rendering, no
    surface viz -- runs at ~2100 Hz on this machine (0.48ms/iteration
    budget). Writing all _RIPPLE_GRID_N**2 vertices into the mesh via
    individual ChVector3d assignments costs ~0.33ms per 144 vertices,
    extrapolating to ~0.6ms at 256 -- comparable to or larger than the
    entire existing per-iteration budget on its own. Doing that every
    iteration would roughly double per-iteration cost. So step_wave() (the
    numpy-only wave simulation, microseconds regardless of grid size) runs
    every physics iteration at the real, numerically-correct 1ms timestep;
    sync_mesh() (the expensive per-vertex PyChrono write-back) is meant to
    be called only every _RIPPLE_MESH_SYNC_EVERY_N_STEPS-th iteration by the
    caller -- an effective ~525 Hz visual refresh at the default throttle,
    far above the ~30-60 Hz a human eye needs to perceive smooth motion.
    """

    def __init__(self, system: chrono.ChSystem, tank_config: TankConfig, fluid_config: "FluidConfig") -> None:
        interior_length = tank_config.length - 2.0 * tank_config.wall_thickness
        interior_width = tank_config.width - 2.0 * tank_config.wall_thickness
        n = _RIPPLE_GRID_N
        self._n = n
        self._x_coords = np.linspace(-interior_length / 2.0, interior_length / 2.0, n)
        self._z_coords = np.linspace(-interior_width / 2.0, interior_width / 2.0, n)
        # Ripple height only (meters, relative to the fill line) -- the base
        # tilt is added on top of this in sync_mesh(), not folded in here,
        # so the wave simulation itself stays a clean, separate model.
        self._h_prev = np.zeros((n, n))
        self._h_curr = np.zeros((n, n))

        self.mesh = chrono.ChTriangleMeshConnected()
        verts = self.mesh.GetCoordsVertices()
        for x in self._x_coords:
            for z in self._z_coords:
                verts.append(chrono.ChVector3d(x, 0.0, z))
        # Chrono renamed this method between builds (confirmed directly on
        # this machine: conda-forge's pychrono 10.0.0 has GetIndicesVertexes,
        # the from-source Chrono::FSI build has GetIndicesVertices instead --
        # and this module needs the FSI build's pychrono.irrlicht, so it's
        # the FSI build's naming that actually runs here). Tries both so
        # this doesn't silently break again on a different Chrono build.
        indices = (
            self.mesh.GetIndicesVertices()
            if hasattr(self.mesh, "GetIndicesVertices")
            else self.mesh.GetIndicesVertexes()
        )
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = i * n + (j + 1)
                c = (i + 1) * n + j
                d = (i + 1) * n + (j + 1)
                indices.append(chrono.ChVector3i(a, b, c))
                indices.append(chrono.ChVector3i(b, d, c))

        visual_shape = chrono.ChVisualShapeTriangleMesh()
        visual_shape.SetMesh(self.mesh)
        visual_shape.SetMutable(True)
        material = chrono.ChVisualMaterial()
        material.SetDiffuseColor(chrono.ChColor(*_SURFACE_DIFFUSE_COLOR))
        material.SetOpacity(_SURFACE_OPACITY)
        material.SetSpecularColor(chrono.ChColor(*_SURFACE_SPECULAR_COLOR))
        material.SetSpecularExponent(_SURFACE_SPECULAR_EXPONENT)
        visual_shape.AddMaterial(material)

        self.body = chrono.ChBody()
        self.body.SetFixed(True)  # kinematic -- position/rotation set by hand every frame, same as the marker
        self.body.EnableCollision(False)
        self.body.AddVisualShape(visual_shape, chrono.ChFramed())
        system.Add(self.body)

    def step_wave(self, tank_angular_velocity: float) -> None:
        """Advance the decorative ripple by one real physics step (numpy
        only -- cheap regardless of grid size, safe to call every iteration
        even though sync_mesh() below is throttled).

        Classic two-buffer damped wave ("water ripple") scheme: each cell's
        next height is the average of its 4 neighbors minus its OWN value
        from two steps ago, then damped -- a well-known, cheap, visually
        convincing real-time-graphics technique, not a shallow-water/Navier-
        Stokes solver. Zero-gradient (reflective) boundaries. Excitation is
        injected at the two length-wise edges (the walls the tank actually
        tips toward/away from), proportional to and opposite-signed with
        tank_angular_velocity -- a plausible-looking stand-in for "the
        rocking tank agitates the liquid near the walls," not a derivation
        from real sloshing mechanics.
        """
        h = self._h_curr
        laplacian_avg = np.zeros_like(h)
        laplacian_avg[1:-1, 1:-1] = 0.5 * (
            h[:-2, 1:-1] + h[2:, 1:-1] + h[1:-1, :-2] + h[1:-1, 2:]
        )
        h_next = laplacian_avg - self._h_prev
        h_next *= _RIPPLE_DAMPING
        h_next[0, :] = h_next[1, :]
        h_next[-1, :] = h_next[-2, :]
        h_next[:, 0] = h_next[:, 1]
        h_next[:, -1] = h_next[:, -2]
        excitation = _RIPPLE_EXCITATION_GAIN * tank_angular_velocity
        h_next[0, :] += excitation
        h_next[-1, :] -= excitation
        self._h_prev, self._h_curr = self._h_curr, h_next

    def sync_mesh(self, tilt_rad: float) -> None:
        """Push the current height field (base tilt + ripple) into the
        actual PyChrono mesh vertices -- the expensive part; the caller is
        expected to call this only every _RIPPLE_MESH_SYNC_EVERY_N_STEPS-th
        iteration, not every step_wave() call. See this class's own
        docstring for the measured cost/rate this is sized against.

        tilt_rad is folded in as a linear height-shear (tilt_rad * x) rather
        than a rigid-body rotation, since a height field can't rotate as a
        whole the way the old flat slab did -- a small-angle approximation
        of that same rotation (error is small at the existing ~15 deg cap).
        What drives the tilt AMOUNT is unchanged from the original slab
        (_surface_tilt_rad(), the pendulum-swing proxy); only how it's
        expressed geometrically changes here.
        """
        n = self._n
        verts = self.mesh.GetCoordsVertices()
        for i, x in enumerate(self._x_coords):
            shear = tilt_rad * x
            ripple_row = self._h_curr[i]
            row_offset = i * n
            for j in range(n):
                v = verts[row_offset + j]
                verts[row_offset + j] = chrono.ChVector3d(v.x, shear + ripple_row[j], v.z)
        self.mesh.Update()


def _surface_tilt_rad(fluid_com_x: float, pendulum_length: float) -> float:
    """Cosmetic-only proxy for how far the liquid surface should visually tilt.

    fluid_com_x / pendulum_length is exactly sin(theta) of the equivalent
    pendulum's own swing angle (see PendulumBackend.get_state()) -- reading
    an existing, already-bounded ([-1, 1]) ratio, not computing any new
    physics quantity. This is a PHYSICALLY-MOTIVATED PROXY for surface
    slope (treating "the bob has swung toward one side" as "the liquid has
    piled up toward that side"), not an exact free-surface-slope
    calculation derived from first principles -- deliberately simple, for
    rendering only, and should not be read as a higher-fidelity physics
    result than it is.
    """
    normalized = max(-1.0, min(1.0, fluid_com_x / pendulum_length))
    return normalized * _SURFACE_MAX_TILT_RAD


def main() -> None:
    args = _parse_args()

    use_fluid_viz = args.fluid_viz or args.fluid_viz_resonance or args.fluid_viz_interactive

    world_config = WorldConfig()
    # Phase 6 replaces the fixed constraint with the motor joint -- see
    # TankConfig.fixed's own docstring for why this was True through
    # Phase 4-5's own checkpoint.
    tank_config = TankConfig(fixed=False)
    pivot_config = PivotConfig()
    fluid_config = FluidConfig()

    # FluidBuilder.compute_backend_parameters() is pure analytical Housner
    # math (see its own docstring: "no ChSystem or ChBody calls") -- safe to
    # call before any PyChrono object exists, and needed here BEFORE
    # MotorConfig is constructed so --fluid-viz-resonance can read the
    # pendulum's real natural frequency instead of a hardcoded guess.
    fluid_builder = FluidBuilder(tank_config, fluid_config, world_config)
    backend_params = fluid_builder.compute_backend_parameters()

    if args.fluid_viz_resonance:
        motor_config = MotorConfig(frequency_hz=backend_params.natural_frequency_hz)
        logger.info(
            "--fluid-viz-resonance enabled: motor frequency overridden to "
            "%.4f Hz (the pendulum's own natural frequency) for THIS demo "
            "run only. MotorConfig's real default (%.4f Hz) is untouched "
            "and still what training/evaluation/tests use.",
            backend_params.natural_frequency_hz, MotorConfig().frequency_hz,
        )
    else:
        motor_config = MotorConfig()

    system = WorldBuilder(world_config).build()

    tank_builder = TankBuilder(tank_config)
    tank_body = tank_builder.build(system)
    tank_com_local = tank_builder.compute_mass_properties().center_of_mass

    pivot_body = PivotBuilder(pivot_config).build(system)
    motor = MotorBuilder(pivot_config, motor_config).build(system, pivot_body, tank_body)

    # Called exactly once, right after build() -- see
    # MotorBuilder.make_live_angle_function()'s own docstring for why this
    # is a one-time swap, not a mechanism for repeatedly re-wiring the
    # motor. Starting values match MotorConfig's own (already possibly
    # overridden by --fluid-viz-resonance above) -- AUTOMATIC mode below
    # never touches them, so it starts numerically identical to today's
    # scripted motor.
    live_motor_function = None
    mode = _MODE_AUTOMATIC
    if args.fluid_viz_interactive:
        live_motor_function = MotorBuilder.make_live_angle_function(
            motor, motor_config.amplitude_rad, motor_config.frequency_hz
        )
        logger.info(
            "--fluid-viz-interactive enabled: starting in %s mode (identical "
            "to the scripted motor). Console must have keyboard focus -- "
            "press 'm' to toggle MANUAL, then '['/']' for amplitude and "
            "','/'.' for frequency.",
            _MODE_AUTOMATIC,
        )

    if use_fluid_viz:
        ripple_surface = _RippleSurface(system, tank_config, fluid_config)
    else:
        marker_body = _build_marker_body(system)

    fluid_backend = fluid_builder.build()
    fluid_backend.reset()
    pivot_height = backend_params.convective_pivot_height
    pendulum_length = backend_params.pendulum_length

    logger.info("Tank mass: %.4f kg", tank_body.GetMass())
    logger.info("Tank COM (world): %s", tank_body.GetPos())
    if use_fluid_viz:
        logger.info(
            "Fluid surface visualization enabled: rendering a %dx%d deforming "
            "height-field (max base tilt %.1f deg, mesh synced every %d steps) "
            "instead of the marker dot. Cosmetic only -- see _RippleSurface's "
            "own docstring for what's real vs. decorative.",
            _RIPPLE_GRID_N, _RIPPLE_GRID_N, math.degrees(_SURFACE_MAX_TILT_RAD),
            _RIPPLE_MESH_SYNC_EVERY_N_STEPS,
        )

    vis = chronoirr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle("LiquidSloshingRL - Phase 6-7: Motor-Driven Tank + Fluid Backend")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(0.9, 0.6, 0.9), chrono.ChVector3d(0.0, 0.15, 0.0))
    vis.AddTypicalLights()

    step_count = 0
    while vis.Run():
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        if args.fluid_viz_interactive and msvcrt.kbhit():
            key_char = msvcrt.getch().decode("ascii", errors="ignore")
            if key_char == "m":
                mode = _MODE_MANUAL if mode == _MODE_AUTOMATIC else _MODE_AUTOMATIC
                logger.info("Mode -> %s", mode)
            elif mode == _MODE_MANUAL:
                # Value changes only -- never a function-object swap or a
                # mode-toggle side effect. See _LiveSineAngleFunction's own
                # docstring for the resulting (expected) angle jump.
                if key_char == "]":
                    live_motor_function.amplitude_rad = min(
                        _AMPLITUDE_MAX_RAD, live_motor_function.amplitude_rad + _AMPLITUDE_STEP_RAD
                    )
                    logger.info(
                        "amplitude_rad -> %.4f rad (%.2f deg)",
                        live_motor_function.amplitude_rad, math.degrees(live_motor_function.amplitude_rad),
                    )
                elif key_char == "[":
                    live_motor_function.amplitude_rad = max(
                        _AMPLITUDE_MIN_RAD, live_motor_function.amplitude_rad - _AMPLITUDE_STEP_RAD
                    )
                    logger.info(
                        "amplitude_rad -> %.4f rad (%.2f deg)",
                        live_motor_function.amplitude_rad, math.degrees(live_motor_function.amplitude_rad),
                    )
                elif key_char == ".":
                    live_motor_function.frequency_hz = min(
                        _FREQUENCY_MAX_HZ, live_motor_function.frequency_hz + _FREQUENCY_STEP_HZ
                    )
                    logger.info("frequency_hz -> %.4f Hz", live_motor_function.frequency_hz)
                elif key_char == ",":
                    live_motor_function.frequency_hz = max(
                        _FREQUENCY_MIN_HZ, live_motor_function.frequency_hz - _FREQUENCY_STEP_HZ
                    )
                    logger.info("frequency_hz -> %.4f Hz", live_motor_function.frequency_hz)

        system.DoStepDynamics(world_config.time_step)

        tank_motion = MotorBuilder.read_tank_motion(motor)
        fluid_backend.step(world_config.time_step, tank_motion)

        bob_state = fluid_backend.get_state()
        tilt_rad = None
        if use_fluid_viz:
            # The grid's reference position is simply the fill line, centered
            # in x/z -- individual vertex heights (tilt shear + ripple, see
            # _RippleSurface.sync_mesh()) are what actually varies, not this
            # body-level transform. Re-based onto the tank body's own local
            # origin (its composite COM, not the footprint origin) the same
            # way the marker's point is below, before
            # TransformPointLocalToParent applies.
            tankconfig_local_point = (0.0, fluid_config.fill_height, 0.0)
            body_local_point = chrono.ChVector3d(
                tankconfig_local_point[0] - tank_com_local[0],
                tankconfig_local_point[1] - tank_com_local[1],
                tankconfig_local_point[2] - tank_com_local[2],
            )
            ripple_surface.body.SetPos(tank_body.TransformPointLocalToParent(body_local_point))
            # No extra tilt quaternion here (unlike the old rigid slab) --
            # the whole grid just rides along with the tank's own rotation;
            # the tilt itself is baked into vertex heights in sync_mesh().
            ripple_surface.body.SetRot(tank_body.GetRot())
            tilt_rad = _surface_tilt_rad(bob_state.com_position[0], pendulum_length)
            # Cheap (numpy) -- runs every iteration at the real timestep.
            # See _RippleSurface.step_wave()'s own docstring.
            ripple_surface.step_wave(tank_motion[1])
            # Expensive (per-vertex PyChrono write-back) -- throttled. See
            # _RippleSurface's own docstring for the measured cost/rate
            # this throttle is sized against.
            if step_count % _RIPPLE_MESH_SYNC_EVERY_N_STEPS == 0:
                ripple_surface.sync_mesh(tilt_rad)
        else:
            # bob_state.com_position is in the pendulum's OWN frame: origin
            # at its pivot, hanging down as the pendulum swings. That pivot
            # sits at (0, pivot_height, 0) in TankConfig's local frame
            # (footprint center, floor level, y=0) -- but tank_body's own
            # body-local origin is NOT that point, it's the tank's
            # composite center of mass (see tank_builder.py::build, which
            # places SetPos at world_com). So the TankConfig-local point
            # has to be re-based onto the body's own local origin (subtract
            # tank_com_local) before TransformPointLocalToParent will place
            # it correctly.
            tankconfig_local_point = (
                bob_state.com_position[0],
                pivot_height + bob_state.com_position[1],
                bob_state.com_position[2],
            )
            body_local_point = chrono.ChVector3d(
                tankconfig_local_point[0] - tank_com_local[0],
                tankconfig_local_point[1] - tank_com_local[1],
                tankconfig_local_point[2] - tank_com_local[2],
            )
            marker_body.SetPos(tank_body.TransformPointLocalToParent(body_local_point))

        step_count += 1
        if step_count % _LOG_EVERY_N_STEPS == 0:
            message = "t=%.2fs tank_angle=%.4f rad tank_angle_dt=%.4f rad/s slosh_energy=%.6f J" % (
                step_count * world_config.time_step,
                tank_motion[0], tank_motion[1], bob_state.slosh_energy,
            )
            if tilt_rad is not None:
                message += " surface_tilt=%.4f rad (%.2f deg)" % (tilt_rad, math.degrees(tilt_rad))
            if args.fluid_viz_interactive:
                # The minimal readout requested: current mode plus the two
                # control values, alongside tank_angle/slosh_energy already
                # above -- tank_angle/slosh_energy aren't repeated here since
                # they're already in the line.
                message += " mode=%s amplitude_rad=%.4f (%.2f deg) frequency_hz=%.4f" % (
                    mode, live_motor_function.amplitude_rad,
                    math.degrees(live_motor_function.amplitude_rad),
                    live_motor_function.frequency_hz,
                )
            logger.info(message)


if __name__ == "__main__":
    main()
