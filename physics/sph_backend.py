"""
Phase 13-14 -- Chrono::FSI (SPH) fluid backend.

Implements FluidBackend using a real Chrono::FSI particle simulation
instead of PendulumBackend's equivalent-pendulum approximation. Per the
deliberate design decision in the development log Entry 41 (read that entry
before changing this file's architecture):

- Kinematic-puppet coupling (path a), NOT true two-way coupling. The
  tank body here is driven directly by whatever (tank_angle,
  tank_angular_velocity) `step()` is given -- it never solves for the
  tank's own motion under the fluid's reaction force. This matches
  FluidBackend.step(dt, tank_motion)'s own one-way contract exactly (see
  physics/backend_interface.py), which was the ORIGINAL Phase 13 design,
  not something imposed after the fact.
- A validation/comparison backend, NOT a training-time replacement for
  PendulumBackend. Measured FSI performance (the development log Entry 40:
  ~35x slower than real-time with zero RL overhead) rules out anything
  training-loop-adjacent, mirroring why environment/chrono_tank_dynamics.py
  was never wired into training/train_ppo.py. The intended use is: record
  a tank_motion trajectory once (from a policy running against the fast
  surrogate), then replay that SAME trajectory through both
  PendulumBackend and SPHBackend and diff their fluid_com/slosh_energy --
  see scripts/record_fluid_trajectory.py and
  scripts/compare_fluid_backends.py.

Requires the real Chrono::FSI Python bindings (pychrono.fsi), which are
NOT installed in this project's local conda env (confirmed,
the development log Entry 19) -- only on the Colab GPU build described in
the project notes' Phase 13-14 section. This module therefore imports
pychrono.core/pychrono.fsi at module level, same convention
environment/chrono_tank_dynamics.py already uses for pychrono.core (no
internal try/except guard) -- it is Colab-side/script-only by design,
never imported by environment/sloshing_env.py's default construction
path. The caller is responsible for the ctypes.CDLL(RTLD_GLOBAL) preload
sequence documented in the project notes BEFORE importing this module (that
sequence is Colab-session-specific -- different CHRONO_ROOT paths on
every fresh session -- so it does not belong hardcoded into project
source).

Status: this is a first implementation, informed by every finding in
the development log Entries 33-41, but NOT YET RUN against the real FSI
bindings by this module's author (only pychrono.core, the non-FSI
subset, is available locally). The kinematic-puppet driving mechanism
(a ChFunction subclass with GetVal/GetDer both overridden and updated
every internal substep -- see _KinematicDrivenAngleFunction and
SPHBackend.step()) went through two rounds of REAL local verification,
not assumption, using a local, FSI-free rigid-body-only test (see
the development log Entry 42): (1) confirmed ChLinkMotorRotationAngle's
solver genuinely reads an overridden GetDer for the constraint's
velocity, not a numerical finite-difference of GetVal alone; (2) then,
testing over a longer duration matching a real external step() call
(0.02s, not the first round's single 1e-4s substep), found that holding
GetVal constant across that whole span -- the original, simpler design
-- actually corrupts the result: the constraint solver's own
stabilization drags the angle back toward the stale GetVal value even
with a correct GetDer feeding the true velocity (measured: final angle
~200x smaller than the correct velocity-integrated value, reported
velocity reading back as ~0). Fixed by having step() linearly
interpolate angle_rad across every internal step_size substep instead
of setting it once per external call -- confirmed by the same test to
reproduce the exact expected angle and velocity. The FULL combination --
this corrected driving mechanism plus a real FSI fluid scene -- has not
yet been exercised end to end; treat any specific numeric prediction
from an untested SPHBackend run with appropriate caution until
confirmed for real on Colab.

Also unconfirmed: `ChFluidSystemSPH.GetParticleVelocitiesNumpy()` is
assumed to exist by naming symmetry with `GetParticlePositionsNumpy()`
(the latter is real and confirmed working, Entries 36-40) but has never
been directly exercised. `_com_world_velocity()` below falls back to
finite-differencing COM position across steps if that method name turns
out to be wrong, rather than crashing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pychrono.core as chrono
import pychrono.fsi as fsi

from config.simulation_config import FluidConfig, TankConfig
from physics.backend_interface import FluidBackend, FluidState
from utilities.fsi_tank_geometry import tank_box_geometry_for_fsi
from utilities.inertia_utils import compute_composite_mass_properties
from utilities.tank_geometry import tank_component_boxes


class _KinematicDrivenAngleFunction(chrono.ChFunction):
    """theta(t) driven externally by SPHBackend.step(), updated every
    INTERNAL substep (not just once per external step() call) -- NOT a
    closed-form function of time the way builders/motor_builder.py's
    _SineAngleFunction/_LiveSineAngleFunction are. GetVal/GetDer both
    ignore their time argument `x` entirely and return whatever
    angle_rad/angular_velocity_rad_s were last set -- same "ignore x,
    return a live-mutable Python attribute" technique
    _ConstantTorqueFunction already uses for the same reason (an
    externally-driven value, not a formula).

    Both GetVal AND GetDer must be kept genuinely current, not just
    GetDer: a real, direct local test (the development log Entry 42) found
    that ChLinkMotorRotationAngle's constraint stabilization pulls the
    joint's angle back toward a STALE GetVal value over time even with a
    correct GetDer supplying the instantaneous velocity -- holding
    GetVal constant across a full external step() call (e.g. one 0.02s
    RL control step) measurably corrupted the result (final angle ~200x
    smaller than the correct velocity-integrated value, reported
    velocity reading back as ~0 instead of the true value). SPHBackend.step()
    fixes this by linearly interpolating angle_rad across every internal
    step_size substep within one external call, not just setting it once
    at the boundary -- confirmed by the same local test to reproduce the
    exact expected angle and velocity.
    """

    def __init__(self, angle_rad: float = 0.0, angular_velocity_rad_s: float = 0.0) -> None:
        super().__init__()
        self.angle_rad = angle_rad
        self.angular_velocity_rad_s = angular_velocity_rad_s

    def GetVal(self, x: float) -> float:
        return self.angle_rad

    def GetDer(self, x: float) -> float:
        return self.angular_velocity_rad_s

    def Clone(self) -> "chrono.ChFunction":
        return _KinematicDrivenAngleFunction(self.angle_rad, self.angular_velocity_rad_s)


class SPHBackend(FluidBackend):
    """Chrono::FSI-backed fluid model, kinematic-puppet coupled.

    Pivot is assumed fixed at the sim-frame origin, matching
    PivotConfig()'s own default -- this class does not currently accept
    a PivotConfig at all (no test in this project has ever needed a
    non-default pivot location or axis for the FSI tank, so support for
    one isn't built speculatively).

    Axis convention: TankConfig/utilities/tank_geometry.py use the
    project's native (x=length, y=height-up, z=width) frame. Chrono::FSI's
    Construct() requires its third position slot to be floor-based
    (confirmed, the development log Entry 36), so internally this class swaps
    y/z throughout (x_sim=length, y_sim=width, z_sim=height) exactly as
    every prior Colab test in Entries 36-40 did, and swaps back to the
    project's native (x, y) convention in get_state() so callers never
    see the internal swap.
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        initial_spacing: float = 0.01,
        step_size: float = 1.0e-4,
        max_velocity: float = 4.5,
        viscosity: float = 1.0,
        artificial_viscosity: float = 0.03,
        num_bce_layers: int = 4,
    ) -> None:
        # max_velocity/viscosity/artificial_viscosity were hardcoded SPH-
        # parameter literals until 2026-09-18 (an external CFD-advisor
        # review, SPH_Enhancement_Guide_18Sept2026.zip, flagged initial_
        # spacing=0.01 as too coarse for publication-quality rendering --
        # see the development log for the dated entry). Exposed here as opt-in
        # constructor params, same pattern ClosedLoopSPHTank already uses
        # for artificial_viscosity/num_bce_layers/boundary_method, so a
        # future GPU session can A/B test finer resolution in a real,
        # controlled way WITHOUT silently changing this class's default
        # behavior -- every existing local test (tests/test_sph_backend.py)
        # and every historical Colab comparison in the development log ran at
        # these exact default values, so the defaults themselves are left
        # untouched pending a real GPU-verified comparison.
        #
        # num_bce_layers exposed 2026-09-19 (the development log Entry 96, RTX
        # 3090 machine) for the same reason: a real dx=0.005 run crashed
        # with a genuine SPH solver blowup (Error flag intercepted ... from
        # CfdCalcRHS_D) at a fixed step regardless of --step-size, pointing
        # at the ADAMI boundary layer's own physical thickness
        # (num_bce_layers * initial_spacing) shrinking proportionally with
        # finer spacing unless compensated -- exposed to test that directly.
        self._tank_config = tank_config or TankConfig()
        self._fluid_config = fluid_config or FluidConfig()
        self._initial_spacing = initial_spacing
        self._step_size = step_size
        self._max_velocity = max_velocity
        self._viscosity = viscosity
        self._artificial_viscosity = artificial_viscosity
        self._num_bce_layers = num_bce_layers

        mass_props = compute_composite_mass_properties(
            tank_component_boxes(self._tank_config)
        )
        self._tank_mass = mass_props.mass
        com = mass_props.center_of_mass
        self._tank_com_sim = (com[0], com[2], com[1])
        inertia = mass_props.inertia_tensor
        self._tank_inertia_sim = (inertia[0][0], inertia[2][2], inertia[1][1])

        self._system = None
        self._fsi_problem = None
        self._fluid_sys = None
        self._drive_function = None
        self._num_fluid_markers = 0
        self._prev_com_world: Optional[np.ndarray] = None
        self._rest_com_tank_frame = np.zeros(3)

        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Rebuild the entire FSI scene from scratch.

        seed is accepted (not used) purely to satisfy FluidBackend's
        signature -- same convention environment/chrono_tank_dynamics.py's
        ChronoTankDynamics.reset() already uses, for the same reason (no
        stochastic element of its own). Rebuilding from scratch, rather
        than resetting bodies in place, isn't a style preference here the
        way it is for ChronoTankDynamics -- it's close to required: a
        previously-Initialize()'d ChFsiProblemCartesian raises
        `RuntimeError: Expression '!m_is_initialized' returned false` on
        any further configuration call (confirmed directly, hit and
        worked around during Entries 39-40's own development).
        """
        del seed

        sys_mbs = chrono.ChSystemNSC()
        sys_mbs.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        fsi_problem = fsi.ChFsiProblemCartesian(self._initial_spacing, sys_mbs)
        fsi_problem.SetVerbose(False)
        fsi_problem.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.8))
        fsi_problem.SetStepSizeCFD(self._step_size)
        fsi_problem.SetStepsizeMBD(self._step_size)

        fluid_props = fsi.FluidProperties()
        fluid_props.density = self._fluid_config.density
        fluid_props.viscosity = self._viscosity
        fsi_problem.SetCfdSPH(fluid_props)

        sph_params = fsi.SPHParameters()
        sph_params.integration_scheme = fsi.IntegrationScheme_RK2
        sph_params.num_bce_layers = self._num_bce_layers
        sph_params.initial_spacing = self._initial_spacing
        sph_params.d0_multiplier = 1
        sph_params.max_velocity = self._max_velocity
        sph_params.shifting_method = fsi.ShiftingMethod_XSPH
        sph_params.shifting_xsph_eps = 0.5
        sph_params.artificial_viscosity = self._artificial_viscosity
        sph_params.eos_type = fsi.EosType_TAIT
        sph_params.use_consistent_gradient_discretization = False
        sph_params.use_consistent_laplacian_discretization = False
        sph_params.num_proximity_search_steps = 1
        sph_params.use_delta_sph = True
        sph_params.delta_sph_coefficient = 0.1
        sph_params.use_variable_time_step = True
        sph_params.boundary_method = fsi.BoundaryMethod_ADAMI
        sph_params.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_UNILATERAL
        fsi_problem.SetSPHParameters(sph_params)

        geometry = chrono.ChBodyGeometry()
        geometry.materials.append(chrono.ChContactMaterialData())
        for box in tank_box_geometry_for_fsi(self._tank_config):
            center_sim = (box.center[0], box.center[2], box.center[1])
            center_rel_com = chrono.ChVector3d(
                center_sim[0] - self._tank_com_sim[0],
                center_sim[1] - self._tank_com_sim[1],
                center_sim[2] - self._tank_com_sim[2],
            )
            dims_sim = chrono.ChVector3d(
                box.full_dimensions[0], box.full_dimensions[2], box.full_dimensions[1]
            )
            geometry.coll_boxes.append(
                chrono.BoxShape(center_rel_com, chrono.QUNIT, chrono.ChBox(dims_sim))
            )

        tank_body = chrono.ChBody()
        tank_body.SetName("tank")
        # Body's own local origin IS the center of mass -- this binding has
        # no exposed setter to offset COM from the reference origin
        # otherwise (confirmed via dir(), the development log Entry 39).
        tank_body.SetPos(chrono.ChVector3d(*self._tank_com_sim))
        tank_body.SetRot(chrono.QUNIT)
        tank_body.SetMass(self._tank_mass)
        tank_body.SetInertiaXX(chrono.ChVector3d(*self._tank_inertia_sim))
        tank_body.SetFixed(False)
        tank_body.EnableCollision(False)
        sys_mbs.AddBody(tank_body)
        fsi_problem.AddRigidBody(tank_body, geometry, True, True)

        ground = chrono.ChBody()
        ground.SetFixed(True)
        ground.EnableCollision(False)
        sys_mbs.AddBody(ground)

        drive_function = _KinematicDrivenAngleFunction(0.0, 0.0)
        pivot_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.Q_ROTATE_Z_TO_Y)
        motor = chrono.ChLinkMotorRotationAngle()
        motor.Initialize(tank_body, ground, pivot_frame)
        motor.SetAngleFunction(drive_function)
        # Keep the Python-side function object alive -- pybind11 directors
        # can end up holding only a weak/borrowed reference on the C++
        # side (see builders/motor_builder.py's own note on this exact
        # issue).
        motor._motion_function_keepalive = drive_function
        sys_mbs.Add(motor)

        interior_length = self._tank_config.length - 2 * self._tank_config.wall_thickness
        interior_width = self._tank_config.width - 2 * self._tank_config.wall_thickness
        fluid_size = chrono.ChVector3d(
            interior_length, interior_width, self._fluid_config.fill_height
        )
        # position.z must be floor-based (0), NOT centered -- Construct()'s
        # non-obvious axis quirk found and fixed in the development log Entry 36.
        fsi_problem.Construct(fluid_size, chrono.ChVector3d(0, 0, 0), 0)
        fsi_problem.Initialize()

        self._system = sys_mbs
        self._fsi_problem = fsi_problem
        self._fluid_sys = fsi_problem.GetFluidSystemSPH()
        self._drive_function = drive_function
        self._num_fluid_markers = self._fluid_sys.GetNumFluidMarkers()
        self._prev_com_world = None

        # Fluid's own rest-state COM (tank frame, theta=0), recorded as the
        # reference get_state() reports a DISPLACEMENT from -- see
        # get_state()'s own docstring for why this is a displacement, not
        # an absolute position.
        self._rest_com_tank_frame = self._rotate_to_tank_frame(self._com_world_position(), 0.0)

    def step(self, dt: float, tank_motion: Tuple[float, float]) -> None:
        """Advance by dt seconds, driving the tank to tank_motion's target
        angle/angular velocity.

        Subdivides dt into internal step_size increments itself (rather
        than a single DoStepDynamics(dt) call relying on Chrono's own
        internal subdivision), linearly interpolating the drive function's
        angle_rad from its current value to the target across those
        increments -- see _KinematicDrivenAngleFunction's own docstring
        for why: a single large dt (e.g. a full 0.02s RL control step)
        with angle_rad set only once at the boundary was confirmed, via a
        real local test, to corrupt the motor's actual motion (the
        constraint solver drags the angle back toward the stale value
        rather than tracking the given angular velocity).
        """
        target_angle, target_angular_velocity = tank_motion
        start_angle = self._drive_function.angle_rad

        n_substeps = max(1, round(dt / self._step_size))
        substep_dt = dt / n_substeps
        self._drive_function.angular_velocity_rad_s = target_angular_velocity
        for i in range(1, n_substeps + 1):
            frac = i / n_substeps
            self._drive_function.angle_rad = start_angle + (target_angle - start_angle) * frac
            self._fsi_problem.DoStepDynamics(substep_dt)

    def get_state(self) -> FluidState:
        """Returns fluid COM position/velocity as a DISPLACEMENT from the
        fluid's own recorded rest state, in the tank's rotating body
        frame -- not an absolute position. PendulumBackend's own
        com_position is measured from the pendulum's PIVOT (a point with
        no physical meaning for a real fluid centroid), so absolute
        values from the two backends are not directly comparable; their
        response to the same tank_motion trajectory is what
        scripts/compare_fluid_backends.py actually compares.

        com_position/com_velocity's z component is always 0.0, matching
        PendulumBackend's planar model -- real SPH fluid motion is NOT
        planar (there is real out-of-plane, width-axis motion), it is
        simply not reported here, by design, for a like-for-like
        comparison against a model that has no width-axis concept at all.
        """
        tank_angle = self._drive_function.angle_rad
        com_world = self._com_world_position()
        com_velocity_world = self._com_world_velocity(com_world)
        self._prev_com_world = com_world

        com_tank_frame = self._rotate_to_tank_frame(com_world, tank_angle)
        com_velocity_tank_frame = self._rotate_to_tank_frame(com_velocity_world, tank_angle)
        displacement = com_tank_frame - self._rest_com_tank_frame

        x, y = float(displacement[0]), float(displacement[2])
        vx, vy = float(com_velocity_tank_frame[0]), float(com_velocity_tank_frame[2])

        # Bulk-COM kinetic energy proxy -- NOT total fluid internal kinetic
        # energy (per-particle motion relative to the bulk COM isn't
        # included). Chosen to mirror PendulumBackend.get_state()'s own
        # 0.5*m*(vx^2+vy^2) formula exactly, for a like-for-like
        # comparison, not because it's a complete energy accounting.
        slosh_energy = 0.5 * self._fluid_mass() * (vx**2 + vy**2)
        # slosh_potential_energy is deliberately left as None here. A
        # COM-height potential (M*g*dy) is not a valid sloshing energy for
        # this solver: the weakly-compressible SPH fluid's COM drifts
        # downward as it settles and as markers penetrate the ADAMI walls
        # (Entry 64), a numerical artifact (measured up to 41 mm at
        # dx=0.01, shrinking to ~0.6 mm at dx=0.002 -- see
        # scripts/recheck_slosh_energy.py) whose M*g*dy is still 3-40x
        # larger than the real peak slosh kinetic energy at every
        # resolution tested.

        return FluidState(
            com_position=(x, y, 0.0),
            com_velocity=(vx, vy, 0.0),
            slosh_energy=slosh_energy,
        )

    def get_fluid_particle_positions(self) -> np.ndarray:
        """Raw per-particle fluid positions, world sim-frame (x_sim=length,
        y_sim=width, z_sim=height), shape (N, 3) -- NOT part of the
        FluidBackend interface (get_state() is the interface-level
        summary), and NOT rotated into the tank's body frame the way
        get_state()'s COM is. This exists purely for visualization/
        debugging -- e.g. rendering the real particle cloud, which no
        FluidBackend consumer needs but a human trying to actually SEE
        the fluid does. Returns a fresh copy each call (not a live view
        into Chrono's own buffer)."""
        positions = np.asarray(self._fluid_sys.GetParticlePositionsNumpy())
        return positions[: self._num_fluid_markers].copy()

    def apply_disturbance(self, impulse: Tuple[float, float, float]) -> None:
        """Not implemented -- out of scope for this validation-only
        backend, not an oversight. Per the development log Entry 41,
        SPHBackend's intended use replays an already-recorded
        tank_motion trajectory whose disturbance was applied upstream
        (against PendulumBackend/TankDynamicsModel, when that trajectory
        was first produced) -- so nothing in the intended comparison
        workflow ever calls this. A real implementation (nudging FSI
        particle velocities directly) is future work, and would need the
        real particle-velocity-mutation API confirmed on Colab first,
        not guessed here.
        """
        raise NotImplementedError(
            "SPHBackend.apply_disturbance is intentionally not implemented -- "
            "see this method's own docstring and the development log Entry 41."
        )

    @property
    def name(self) -> str:
        return "sph"

    def _fluid_mass(self) -> float:
        """Approximate total fluid mass from marker count * SPH particle
        volume (initial_spacing^3, the standard cubic-lattice SPH particle
        volume) * density -- no direct Chrono API for total fluid mass was
        found or used in any of this project's Colab tests so far."""
        particle_volume = self._initial_spacing ** 3
        return self._num_fluid_markers * particle_volume * self._fluid_config.density

    def _com_world_position(self) -> np.ndarray:
        positions = np.asarray(self._fluid_sys.GetParticlePositionsNumpy())
        return positions[: self._num_fluid_markers].mean(axis=0)

    def _com_world_velocity(self, current_com_world: np.ndarray) -> np.ndarray:
        try:
            velocities = np.asarray(self._fluid_sys.GetParticleVelocitiesNumpy())
            return velocities[: self._num_fluid_markers].mean(axis=0)
        except AttributeError:
            # See this module's own docstring -- GetParticleVelocitiesNumpy's
            # existence is assumed by naming symmetry, not confirmed.
            if self._prev_com_world is None:
                return np.zeros(3)
            return (current_com_world - self._prev_com_world) / self._step_size

    @staticmethod
    def _rotate_to_tank_frame(v: np.ndarray, tank_angle: float) -> np.ndarray:
        """Undo the tank's current rigid rotation (about sim-frame Y, by
        tank_angle) so a world-frame quantity is expressed in the tank's
        own body-fixed frame -- the frame PendulumBackend's theta is
        measured in. Pivot is assumed at the sim-frame origin (see this
        class's own docstring), so only the inverse rotation is needed,
        no translation."""
        cos_t, sin_t = math.cos(tank_angle), math.sin(tank_angle)
        x, y, z = v
        return np.array([x * cos_t - z * sin_t, y, x * sin_t + z * cos_t])
