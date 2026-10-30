"""
Live Irrlicht visualization of the trained PPO policy actively controlling
the real PyChrono tank -- unlike main.py (Phase 6-7's scripted-motor demo)
and unlike evaluation/evaluate_policy.py / scripts/validate_real_dynamics.py
(headless, numbers only, no window), this is the first script in the
project that shows the trained agent's live torque decisions in a real,
watchable window.

Reproduces the EXACT control sequencing environment/sloshing_env.py's
SloshingSuppressionEnv.step()/reset() use, not an approximation:
  - One action (torque) is decided per control_dt (0.02s / 50 Hz), then
    held CONSTANT across every physics_dt (0.001s) substep within that
    control step -- substeps_per_control_step (20) of them -- matching
    step()'s own `for _ in range(substeps_per_control_step)` loop exactly.
  - The observation fed to the NEXT action decision is built from the
    state AFTER all of the current control step's substeps have run,
    matching step()'s own ordering (torque decided from the OLD
    observation, new observation built only at the end).
  - A reset-time disturbance impulse is applied once at startup, the same
    way SloshingSuppressionEnv.reset() does, so there is real sloshing for
    the agent to react to instead of a trivially flat starting fluid.

Uses the REAL ChSystem-backed tank (builders/, MotorKind.TORQUE -- the
same construction environment/chrono_tank_dynamics.py::ChronoTankDynamics
uses), not the fast analytic surrogate the model was originally trained
against -- so what you see here is the same kind of real-physics gap
already characterized in the development log Entries 10-11
(scripts/validate_real_dynamics.py), not the training-time behavior
exactly. PendulumBackend (not SPHBackend -- too slow for a live 1kHz loop,
see the development log Entry 41) still provides the fluid observation/reward
state and the visualized marker, same as main.py.

Deliberately simple, first-cut scope, NOT shared with main.py's code
(same "don't cross-import a __main__-oriented script's private helpers"
convention scripts/record_fluid_trajectory.py already established): the
--fluid-viz ripple surface below is a direct, deliberate DUPLICATE of
main.py's own `_RippleSurface`/`_surface_tilt_rad` (same reasoning,
same physically-motivated-proxy caveats -- see that class's own
docstring, copied verbatim below), not an import. --fluid-viz-resonance
and --fluid-viz-interactive are NOT ported here (no scripted motor to
override/retune in this script -- the motor is TORQUE-driven by the
policy, not ANGLE-driven by a sine function). No auto-reset on episode
termination/truncation -- the physics and policy just keep running
continuously past the trained 500-step/10s horizon; this is an accepted
simplification for a live demo, not a bug (the policy was never trained
to expect this, so behavior past 10s is a reasonable extrapolation, not
a validated result).

Run:
    python main_trained_policy.py
    python main_trained_policy.py models/ppo_sloshing_realphysics.zip
    python main_trained_policy.py --fluid-viz
"""

import argparse
import logging
import math

import numpy as np
import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from builders.fluid_builder import FluidBuilder
from builders.motor_builder import MotorBuilder
from builders.pivot_builder import PivotBuilder
from builders.tank_builder import TankBuilder
from builders.world_builder import WorldBuilder
from config.simulation_config import (
    EnvironmentConfig,
    FluidConfig,
    MotorConfig,
    MotorKind,
    PivotConfig,
    TankConfig,
    WorldConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MARKER_RADIUS = 0.012
_MARKER_COLOR = (0.95, 0.35, 0.15)
_LOG_EVERY_N_CONTROL_STEPS = 10
_RESET_DISTURBANCE_IMPULSE = (0.02, 0.0, 0.0)  # N*s -- max magnitude from
# EnvironmentConfig.reset_disturbance_impulse_range's own upper bound, a
# fixed (not random) direction/magnitude so the demo is reproducible.

# --fluid-viz cosmetic constants -- identical values to main.py's own, see
# that module's docstring/_RippleSurface for the full reasoning (grid
# size vs. measured performance budget, damping/excitation choices, etc).
_SURFACE_DIFFUSE_COLOR = (0.15, 0.45, 0.55)
_SURFACE_OPACITY = 0.55
_SURFACE_SPECULAR_COLOR = (0.9, 0.95, 1.0)
_SURFACE_SPECULAR_EXPONENT = 0.7
_SURFACE_MAX_TILT_RAD = math.radians(15.0)
_RIPPLE_GRID_N = 16
_RIPPLE_MESH_SYNC_EVERY_N_STEPS = 4
_RIPPLE_DAMPING = 0.999
_RIPPLE_EXCITATION_GAIN = 0.002


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path", nargs="?", default="models/ppo_sloshing",
        help="Saved PPO model path (with or without .zip). Defaults to the base surrogate-trained policy.",
    )
    parser.add_argument(
        "--vecnormalize-path", default=None,
        help="Defaults to '<model_path>_vecnormalize.pkl' if it exists.",
    )
    parser.add_argument(
        "--fluid-viz",
        action="store_true",
        help="Render the sloshing fluid as a deforming ripple surface instead of the default marker dot -- same cosmetic model as main.py's --fluid-viz. See this module's own docstring.",
    )
    return parser.parse_args()


def _build_marker_body(system: chrono.ChSystem) -> chrono.ChBody:
    marker = chrono.ChBody()
    marker.SetFixed(True)
    marker.EnableCollision(False)
    visual_shape = chrono.ChVisualShapeSphere(_MARKER_RADIUS)
    visual_shape.SetColor(chrono.ChColor(*_MARKER_COLOR))
    marker.AddVisualShape(visual_shape, chrono.ChFramed())
    system.Add(marker)
    return marker


class _RippleSurface:
    """Verbatim duplicate of main.py's own `_RippleSurface` -- see that
    class's docstring for the full real-vs-decorative explanation
    (physically-motivated base tilt proxy, purely illustrative ripple
    pattern, one-way coupled, never feeds back into the real physics).
    Duplicated rather than imported per this module's own docstring."""

    def __init__(self, system: chrono.ChSystem, tank_config: TankConfig, fluid_config: "FluidConfig") -> None:
        interior_length = tank_config.length - 2.0 * tank_config.wall_thickness
        interior_width = tank_config.width - 2.0 * tank_config.wall_thickness
        n = _RIPPLE_GRID_N
        self._n = n
        self._x_coords = np.linspace(-interior_length / 2.0, interior_length / 2.0, n)
        self._z_coords = np.linspace(-interior_width / 2.0, interior_width / 2.0, n)
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
        self.body.SetFixed(True)
        self.body.EnableCollision(False)
        self.body.AddVisualShape(visual_shape, chrono.ChFramed())
        system.Add(self.body)

    def step_wave(self, tank_angular_velocity: float) -> None:
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
    """Verbatim duplicate of main.py's own `_surface_tilt_rad` -- see that
    function's docstring for why this is a proxy, not exact physics."""
    normalized = max(-1.0, min(1.0, fluid_com_x / pendulum_length))
    return normalized * _SURFACE_MAX_TILT_RAD


def _load_normalizer(vecnormalize_path) -> "VecNormalize | None":
    if vecnormalize_path is None:
        return None
    import environment  # noqa: F401 -- gym.register() side effect
    import gymnasium as gym

    throwaway = DummyVecEnv([lambda: gym.make("LiquidSloshingRL/SloshingSuppression-v0")])
    normalizer = VecNormalize.load(vecnormalize_path, throwaway)
    normalizer.training = False
    return normalizer


def main() -> None:
    args = _parse_args()

    import os

    vecnormalize_path = args.vecnormalize_path
    if vecnormalize_path is None:
        default_stats_path = args.model_path.rstrip(".zip") + "_vecnormalize.pkl"
        if os.path.exists(default_stats_path):
            vecnormalize_path = default_stats_path
            logger.info("Found VecNormalize stats at %s, using them.", vecnormalize_path)

    model = PPO.load(args.model_path)
    normalizer = _load_normalizer(vecnormalize_path)
    logger.info("Loaded policy from %s", args.model_path)

    world_config = WorldConfig()
    tank_config = TankConfig(fixed=False)
    pivot_config = PivotConfig()
    fluid_config = FluidConfig()
    env_config = EnvironmentConfig()
    motor_config = MotorConfig(kind=MotorKind.TORQUE)

    fluid_builder = FluidBuilder(tank_config, fluid_config, world_config)
    backend_params = fluid_builder.compute_backend_parameters()
    pivot_height = backend_params.convective_pivot_height
    pendulum_length = backend_params.pendulum_length

    system = WorldBuilder(world_config).build()
    tank_builder = TankBuilder(tank_config)
    tank_body = tank_builder.build(system)
    tank_com_local = tank_builder.compute_mass_properties().center_of_mass

    pivot_body = PivotBuilder(pivot_config).build(system)
    motor = MotorBuilder(pivot_config, motor_config).build(system, pivot_body, tank_body)

    if args.fluid_viz:
        ripple_surface = _RippleSurface(system, tank_config, fluid_config)
    else:
        marker_body = _build_marker_body(system)

    fluid_backend = fluid_builder.build()
    fluid_backend.reset()
    fluid_backend.apply_disturbance(_RESET_DISTURBANCE_IMPULSE)
    logger.info(
        "Applied startup disturbance impulse %s N*s (fixed, matching "
        "EnvironmentConfig.reset_disturbance_impulse_range's max magnitude) "
        "so there is real sloshing for the policy to react to.",
        _RESET_DISTURBANCE_IMPULSE,
    )

    logger.info("Tank mass: %.4f kg", tank_body.GetMass())

    vis = chronoirr.ChVisualSystemIrrlicht()
    vis.AttachSystem(system)
    vis.SetWindowSize(1024, 768)
    vis.SetWindowTitle("LiquidSloshingRL - Trained Policy, Live Control")
    vis.Initialize()
    vis.AddSkyBox()
    vis.AddCamera(chrono.ChVector3d(0.9, 0.6, 0.9), chrono.ChVector3d(0.0, 0.15, 0.0))
    vis.AddTypicalLights()

    def build_observation(tank_angle: float, tank_angular_velocity: float) -> np.ndarray:
        fluid_state = fluid_backend.get_state()
        return np.array(
            [
                tank_angle, tank_angular_velocity,
                fluid_state.com_position[0], fluid_state.com_position[1],
                fluid_state.com_velocity[0], fluid_state.com_velocity[1],
                fluid_state.slosh_energy,
            ],
            dtype=np.float32,
        )

    def update_marker(fluid_state) -> None:
        tankconfig_local_point = (
            fluid_state.com_position[0],
            pivot_height + fluid_state.com_position[1],
            fluid_state.com_position[2],
        )
        body_local_point = chrono.ChVector3d(
            tankconfig_local_point[0] - tank_com_local[0],
            tankconfig_local_point[1] - tank_com_local[1],
            tankconfig_local_point[2] - tank_com_local[2],
        )
        marker_body.SetPos(tank_body.TransformPointLocalToParent(body_local_point))

    def update_ripple(fluid_state, tank_angular_velocity: float, physics_step: int) -> None:
        # Same positioning logic as main.py: the grid's reference point is
        # the fill line, centered in x/z, re-based onto the tank body's own
        # local origin (its composite COM) before TransformPointLocalToParent.
        tankconfig_local_point = (0.0, fluid_config.fill_height, 0.0)
        body_local_point = chrono.ChVector3d(
            tankconfig_local_point[0] - tank_com_local[0],
            tankconfig_local_point[1] - tank_com_local[1],
            tankconfig_local_point[2] - tank_com_local[2],
        )
        ripple_surface.body.SetPos(tank_body.TransformPointLocalToParent(body_local_point))
        ripple_surface.body.SetRot(tank_body.GetRot())
        tilt_rad = _surface_tilt_rad(fluid_state.com_position[0], pendulum_length)
        ripple_surface.step_wave(tank_angular_velocity)
        if physics_step % _RIPPLE_MESH_SYNC_EVERY_N_STEPS == 0:
            ripple_surface.sync_mesh(tilt_rad)

    initial_tank_motion = MotorBuilder.read_tank_motion(motor)
    obs = build_observation(*initial_tank_motion)

    control_step = 0
    physics_step = 0
    while vis.Run():
        model_input = normalizer.normalize_obs(obs) if normalizer is not None else obs
        action, _ = model.predict(model_input, deterministic=True)
        normalized_action = float(np.clip(action, -1.0, 1.0).reshape(-1)[0])
        torque_nm = normalized_action * env_config.max_motor_torque_nm
        MotorBuilder.set_commanded_torque(motor, torque_nm)

        for _ in range(env_config.substeps_per_control_step):
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

            system.DoStepDynamics(env_config.physics_dt)
            tank_motion = MotorBuilder.read_tank_motion(motor)
            fluid_backend.step(env_config.physics_dt, tank_motion)
            if args.fluid_viz:
                update_ripple(fluid_backend.get_state(), tank_motion[1], physics_step)
            else:
                update_marker(fluid_backend.get_state())
            physics_step += 1

        control_step += 1
        fluid_state = fluid_backend.get_state()
        obs = build_observation(*tank_motion)

        if control_step % _LOG_EVERY_N_CONTROL_STEPS == 0:
            logger.info(
                "t=%.2fs tank_angle=%.4f rad torque=%.4f N*m slosh_energy=%.6f J",
                control_step * env_config.control_dt, tank_motion[0], torque_nm,
                fluid_state.slosh_energy,
            )


if __name__ == "__main__":
    main()
