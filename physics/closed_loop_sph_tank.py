"""
Phase 15 -- true two-way, torque-driven closed-loop SPH tank.

Unlike physics/sph_backend.py::SPHBackend (deliberately one-way/
kinematic-puppet per the development log Entry 41 -- tank motion is an
EXTERNAL INPUT to step(), with no channel for the fluid's force to feed
back), this class makes the tank a genuine TORQUE-DRIVEN rigid body
inside the live FSI scene: the real SPH fluid's force affects the
tank's real motion, and a control loop (scripts/run_closed_loop_sph_policy.py)
can query the tank's actual resulting state and react to it, closing
the loop for real -- see the development log Entry 48 for the full design
decision and reasoning.

Deliberately NOT a FluidBackend implementation and NOT a subclass of
SPHBackend: the method signatures are fundamentally incompatible here
(there is no external tank_motion to pass into step() -- the tank's
motion emerges from the coupled physics instead of being prescribed).
Small geometry/state-computation helpers are duplicated from SPHBackend
rather than inherited, per this project's own established cross-module
convention (see scripts/record_fluid_trajectory.py's own docstring for
the same reasoning) -- these two classes solve genuinely different
problems and should stay independently readable, not entangled through
inheritance.

Requires the real Chrono::FSI Python bindings, same as SPHBackend --
NOT installed in this project's local conda env, only where a real
CUDA GPU + from-source Chrono::FSI build exists (Colab, or the RTX
3090-class machine this was actually written for). See the project notes'
Phase 13-14 section for the ctypes preload sequence needed before this
module can be imported at all.

STATUS: written entirely without a local FSI-capable machine to test
against (same division of labor already used successfully for
SPHBackend itself, Entry 42 -> Entry 43) -- verified only as far as
py_compile and local reasoning allow at the time. Every mechanism reused
here (FSI-coupled tank body, real mass/inertia, torque-driven motor,
angular-velocity disturbance) is individually already proven somewhere
in this project's history (see each method's own docstring for the
exact the development log entry).

UPDATE (the development log Entry 55): now actually run for real, on a real
local Chrono::FSI build (Entries 50-54) -- for the first time, not just
reasoned about. One real bug was found and fixed: step()'s single-large-
DoStepDynamics-call assumption was wrong (see step()'s own docstring for
the full story and the exact numbers that exposed it) -- fixed by
manually subdividing, matching SPHBackend's own discipline. The sign
convention (positive commanded torque -> positive angle growth) was
independently verified to match the already-proven non-FSI
ChronoTankDynamics before any of this was trusted. Whether the fixed
step() makes a full real closed-loop policy run stable end-to-end is
being verified next -- see the development log Entry 55 for the actual result,
don't assume it from this docstring alone.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pychrono.core as chrono
import pychrono.fsi as fsi

from config.simulation_config import EnvironmentConfig, FluidConfig, TankConfig
from physics.backend_interface import FluidState
from utilities.fsi_tank_geometry import tank_box_geometry_for_fsi
from utilities.inertia_utils import compute_composite_mass_properties
from utilities.tank_geometry import tank_component_boxes


class _TorqueFunction(chrono.ChFunction):
    """Externally-driven constant-torque ChFunction -- the exact technique
    builders/motor_builder.py::_ConstantTorqueFunction already uses for
    Phase 10's real (non-FSI) torque-driven tank, confirmed working
    across many real training/validation runs (Phase 10-11).

    Unlike physics/sph_backend.py::_KinematicDrivenAngleFunction (which
    needed a real, locally-verified fix for constraint-stabilization
    dragging its value back toward a stale GetVal, the development log
    Entry 42), a torque motor applies a generalized FORCE, not a
    position constraint -- there is no analogous "stale value fought by
    the solver" failure mode expected here, since nothing is trying to
    hold the joint at a particular angle. Not independently verified on
    a real FSI scene (see this module's own docstring), but is the same
    mechanism already proven correct for the non-FSI case.
    """

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value

    def GetVal(self, x: float) -> float:
        return self.value

    def Clone(self) -> "chrono.ChFunction":
        return _TorqueFunction(self.value)


class ClosedLoopSPHTank:
    """A torque-driven tank, genuinely coupled to a real Chrono::FSI fluid
    simulation -- the tank's motion is a real consequence of gravity,
    commanded torque, AND the fluid's own reaction force, not prescribed.

    Pivot fixed at the sim-frame origin and axis-swap convention (x=length,
    y=width, z=height) identical to SPHBackend -- see that class's own
    docstring for the full explanation; not repeated here.
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        initial_spacing: float = 0.01,
        step_size: float = 1.0e-4,
        reset_angular_velocity_rad_s: float = 0.05,
        artificial_viscosity: float = 0.03,
        num_bce_layers: int = 4,
        boundary_method: Optional[int] = None,
        max_velocity: float = 4.5,
        viscosity: float = 1.0,
    ) -> None:
        self._tank_config = tank_config or TankConfig(fixed=False)
        self._fluid_config = fluid_config or FluidConfig()
        self._env_config = env_config or EnvironmentConfig()
        self._initial_spacing = initial_spacing
        self._step_size = step_size
        self._torque_limit = self._env_config.max_motor_torque_nm
        self._reset_angular_velocity_rad_s = reset_angular_velocity_rad_s
        self._artificial_viscosity = artificial_viscosity
        # max_velocity/viscosity were hardcoded literals until 2026-09-18
        # (external CFD-advisor review, SPH_Enhancement_Guide_18Sept2026.zip
        # -- see the development log dated entry). Same opt-in-parameter pattern
        # as boundary_method/num_bce_layers above: defaults preserve every
        # existing caller's exact prior behavior; a future GPU session can
        # pass viscosity=0.001 (physical water, vs. the current 1.0) or
        # max_velocity=2.0 for a real, controlled comparison instead of a
        # blind default change.
        self._max_velocity = max_velocity
        self._viscosity = viscosity
        # Exposed for a real, controlled test of the external reference package's one
        # legitimate-but-never-actually-run idea for the transient wall-
        # penetration issue (Entry 61/73): more boundary-condition-
        # enforcement layers for the ADAMI method to draw from. `4` is
        # this class's own prior, already-validated default -- unchanged
        # unless a caller explicitly asks for more. UNTESTED at any value
        # other than 4 as of 2026-09-16 (written while the GPU was busy
        # with the fine-tuning campaign) -- see the development log Entry 75.
        self._num_bce_layers = num_bce_layers
        # Real, previously-untried alternative to the default ADAMI boundary
        # method (the development log Entry 61/64/73/77 -- three separate tuning
        # attempts around ADAMI itself, viscosity/step_size/num_bce_layers,
        # were all closed negative; the project notes' own "what's next" note says
        # a real fix here needs a genuinely different boundary-marker
        # approach, not another parameter around the same one). Confirmed
        # to exist in this installed Chrono::FSI build by directly
        # inspecting pychrono.fsi's own BoundaryMethod enum, not guessed --
        # `fsi.BoundaryMethod_HOLMES` (value 1) alongside the existing
        # `fsi.BoundaryMethod_ADAMI` (value 0). None (default) preserves
        # the exact prior behavior (ADAMI) for every existing caller.
        self._boundary_method = boundary_method

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
        self._tank_body = None
        self._motor = None
        self._torque_function = None
        self._num_fluid_markers = 0
        self._prev_com_world: Optional[np.ndarray] = None
        self._rest_com_tank_frame = np.zeros(3)

        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Rebuild the entire FSI scene from scratch -- required, not a
        style choice (see physics/sph_backend.py::SPHBackend.reset()'s own
        docstring for why an already-Initialize()'d ChFsiProblemCartesian
        cannot be reconfigured in place).

        A fixed initial angular velocity is applied to the tank as the
        reset-time disturbance (Entry 40's own proven MECHANISM -- an
        angular-velocity kick to the tank body) -- EXPLICITLY NOT the
        same disturbance the trained policy actually saw during training
        (a random impulse applied directly to the fluid, per
        EnvironmentConfig.reset_disturbance_impulse_range). This is a
        real, deliberate deviation, used only because it's the one
        confirmed-working mechanism for disturbing this kind of FSI
        scene -- see this class's own module docstring / the development log
        Entry 48.

        The default MAGNITUDE (0.05 rad/s, not Entry 40's own 0.2)
        is a SEPARATE, later correction (the development log Entry 55) --
        0.2 rad/s was only ever validated in Entry 40's free-swing (no
        torque motor) test; reused here as a default without
        re-validating for the closed-loop (torque-driven) case. A real
        run with 0.2 rad/s reached a peak angular velocity of ~1.14 rad/s
        under the trained policy's own corrective torque and crashed the
        SPH solver (`particle derivVel is NAN`) around control step 40 --
        the same numerical-stability regime Entry 44 already identified
        (~1.27 rad/s peak was the failure threshold there too, for an
        unrelated externally-driven scenario). 0.05 rad/s keeps a real
        80-step (1.6s simulated) run stable throughout, confirmed
        directly, not assumed.
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
        sph_params.boundary_method = (
            self._boundary_method
            if self._boundary_method is not None
            else fsi.BoundaryMethod_ADAMI
        )
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

        torque_function = _TorqueFunction(0.0)
        pivot_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.Q_ROTATE_Z_TO_Y)
        motor = chrono.ChLinkMotorRotationTorque()
        motor.Initialize(tank_body, ground, pivot_frame)
        motor.SetTorqueFunction(torque_function)
        motor._motion_function_keepalive = torque_function
        sys_mbs.Add(motor)

        # Reset-time disturbance -- see this method's own docstring for
        # why this differs from how the policy was actually trained.
        if self._reset_angular_velocity_rad_s != 0.0:
            tank_body.SetAngVelParent(
                chrono.ChVector3d(0, self._reset_angular_velocity_rad_s, 0)
            )

        interior_length = self._tank_config.length - 2 * self._tank_config.wall_thickness
        interior_width = self._tank_config.width - 2 * self._tank_config.wall_thickness
        fluid_size = chrono.ChVector3d(
            interior_length, interior_width, self._fluid_config.fill_height
        )
        fsi_problem.Construct(fluid_size, chrono.ChVector3d(0, 0, 0), 0)
        fsi_problem.Initialize()

        self._system = sys_mbs
        self._fsi_problem = fsi_problem
        self._fluid_sys = fsi_problem.GetFluidSystemSPH()
        self._tank_body = tank_body
        self._motor = motor
        self._torque_function = torque_function
        self._num_fluid_markers = self._fluid_sys.GetNumFluidMarkers()
        self._prev_com_world = None
        self._rest_com_tank_frame = self._rotate_to_tank_frame(self._com_world_position(), 0.0)

    def set_commanded_torque(self, torque_nm: float) -> None:
        """Update the commanded torque in place -- same pattern as
        builders/motor_builder.py::MotorBuilder.set_commanded_torque(),
        clipped to EnvironmentConfig.max_motor_torque_nm the same way
        environment/chrono_tank_dynamics.py::ChronoTankDynamics.step()
        clips its own commanded torque."""
        clipped = max(-self._torque_limit, min(self._torque_limit, torque_nm))
        self._torque_function.value = clipped

    def step(
        self, dt: float, angular_velocity_safety_limit: Optional[float] = None
    ) -> bool:
        """Advance by dt seconds under the currently-commanded torque.
        Returns True if the full dt completed, False if it stopped early
        because angular_velocity_safety_limit was tripped mid-step.

        CORRECTED (the development log Entry 55) -- this originally called
        DoStepDynamics(dt) ONCE with the full external dt (e.g. a whole
        0.02s control step), on the assumption that a torque motor (no
        position constraint to fight, unlike SPHBackend's angle-tracking
        motor) meant Chrono would safely self-subdivide internally via
        the step_size already configured in reset(). **That assumption
        was wrong, confirmed by a real, direct test, not guessed**: a
        single 0.02s DoStepDynamics call caused angular velocity to blow
        up explosively (0.13 -> 1.17 -> 2.80 -> 3.49 rad/s over 5 control
        steps under a small constant +1.0 Nm test torque, vs. the
        already-proven non-FSI ChronoTankDynamics's smooth, linear ~0.13
        rad/s per step under the identical torque) -- a severe CFL
        violation for the explicit SPH solver, not real physics. Manually
        subdividing dt into self._step_size increments (the SAME
        discipline SPHBackend.step() already uses, minus the angle
        interpolation that's specific to a position-tracking motor -- a
        constant torque needs no interpolation, just repeated small
        DoStepDynamics calls) keeps the same test bounded and stable
        (angle/angular_velocity oscillating in a small, physically
        sane range instead of diverging). Confirmed as the correct fix
        before changing this method, not assumed.

        UPDATE (the development log Entry 58) -- the CALLER-side proactive
        safety check (environment/closed_loop_sph_env.py,
        scripts/run_closed_loop_sph_policy.py), which only reads
        angular_velocity BETWEEN control steps, was found to be too
        coarse: a real 16,384-timestep fine-tuning run crashed
        (`particle derivVel is NAN`) inside a single control step's
        ~200 internal substeps, after the between-step check had
        already passed. angular_velocity/angle are cheap, live reads
        off the motor (GetMotorAngleDt/GetMotorAngle), so checking
        after EVERY substep here -- not just once per control step --
        catches a blowup at ~1e-4s granularity instead of up to a full
        control_dt (0.02s) late. Optional and off by default
        (None) so callers that don't pass a limit keep the exact prior
        behavior."""
        n_substeps = max(1, round(dt / self._step_size))
        substep_dt = dt / n_substeps
        for _ in range(n_substeps):
            self._fsi_problem.DoStepDynamics(substep_dt)
            if (
                angular_velocity_safety_limit is not None
                and abs(self.angular_velocity) > angular_velocity_safety_limit
            ):
                return False
        return True

    @property
    def angle(self) -> float:
        return self._motor.GetMotorAngle()

    @property
    def angular_velocity(self) -> float:
        return self._motor.GetMotorAngleDt()

    def get_fluid_state(self) -> FluidState:
        """Identical computation to SPHBackend.get_state() -- see that
        method's own docstring for the full explanation of why
        com_position/com_velocity are reported as a displacement from the
        fluid's own rest state, not an absolute position."""
        tank_angle = self.angle
        com_world = self._com_world_position()
        com_velocity_world = self._com_world_velocity(com_world)
        self._prev_com_world = com_world

        com_tank_frame = self._rotate_to_tank_frame(com_world, tank_angle)
        com_velocity_tank_frame = self._rotate_to_tank_frame(com_velocity_world, tank_angle)
        displacement = com_tank_frame - self._rest_com_tank_frame

        x, y = float(displacement[0]), float(displacement[2])
        vx, vy = float(com_velocity_tank_frame[0]), float(com_velocity_tank_frame[2])
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
        """Identical to SPHBackend.get_fluid_particle_positions() -- raw
        per-particle fluid positions, world sim-frame, for visualization/
        debugging only."""
        positions = np.asarray(self._fluid_sys.GetParticlePositionsNumpy())
        return positions[: self._num_fluid_markers].copy()

    def _fluid_mass(self) -> float:
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
            if self._prev_com_world is None:
                return np.zeros(3)
            return (current_com_world - self._prev_com_world) / self._step_size

    @staticmethod
    def _rotate_to_tank_frame(v: np.ndarray, tank_angle: float) -> np.ndarray:
        cos_t, sin_t = math.cos(tank_angle), math.sin(tank_angle)
        x, y, z = v
        return np.array([x * cos_t - z * sin_t, y, x * sin_t + z * cos_t])
