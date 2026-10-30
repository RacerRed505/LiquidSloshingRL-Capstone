"""LQR baseline controller, linearized from this project's REAL dynamics.

Unlike an independently-invented tank+pendulum model, the A/B matrices
here are built directly from:
  - environment/tank_dynamics.py::tank_moment_of_inertia_about_pivot()
    (the exact same analytical inertia builders/tank_builder.py uses),
  - config/simulation_config.py::EnvironmentConfig.pivot_damping_coefficient
    (the exact damping TankDynamicsModel integrates), and
  - utilities/housner_utils.py::compute_housner_parameters() (the exact
    pendulum_length/natural_frequency FluidBuilder hands to
    PendulumBackend).

So this is a linearization of the SAME surrogate physics
environment/sloshing_env.py::SloshingSuppressionEnv actually trains
against by default -- not a parallel, disconnected reimplementation --
which is what makes a benchmark comparison against a real trained PPO
policy (see benchmarks/run_controller_benchmarks.py) meaningful rather
than a comparison against a strawman.

Model (small-angle, matching physics/pendulum_backend.py's own dynamics
exactly -- see that module for why gravity/coupling take the form they
do):
    theta_ddot = (u - b_tank * theta_dot) / I_tank
    phi_ddot   = -(g / L_p) * phi - c_pendulum * phi_dot - theta_dot

State x = [theta, theta_dot, phi, phi_dot], input u = commanded torque
(N*m). Note this is a ONE-WAY coupling (tank forces the pendulum, the
pendulum does not react back on the tank) because that is exactly what
TankDynamicsModel.step()/PendulumBackend.step() actually implement (see
the development log's own discussion of this as a documented first-pass
approximation, not something this controller module should silently
"fix" by inventing a different, untested plant model).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from config.simulation_config import (
    EnvironmentConfig,
    FluidConfig,
    PivotConfig,
    TankConfig,
    WorldConfig,
)
from environment.tank_dynamics import tank_moment_of_inertia_about_pivot
from utilities.housner_utils import compute_housner_parameters


def build_state_space(
    tank_config: Optional[TankConfig] = None,
    pivot_config: Optional[PivotConfig] = None,
    fluid_config: Optional[FluidConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
    world_config: Optional[WorldConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the continuous-time (A, B) pair for x = [theta, theta_dot, phi, phi_dot].

    Every numeric input is derived from this project's real config
    dataclasses (defaults matching main.py/training's own defaults if
    none are passed), not invented tank/fluid numbers.
    """
    tank_config = tank_config or TankConfig(fixed=False)
    pivot_config = pivot_config or PivotConfig()
    fluid_config = fluid_config or FluidConfig()
    env_config = env_config or EnvironmentConfig()
    world_config = world_config or WorldConfig()

    inertia = tank_moment_of_inertia_about_pivot(tank_config, pivot_config)
    b_tank = env_config.pivot_damping_coefficient

    gravity = abs(world_config.gravity[1])
    housner = compute_housner_parameters(
        tank_length=tank_config.length,
        tank_breadth=tank_config.width,
        fill_height=fluid_config.fill_height,
        fluid_density=fluid_config.density,
        gravity=gravity,
    )
    pendulum_length = housner.pendulum_length
    # Same 2*zeta*omega form builders/fluid_builder.py::FluidBuilder.build()
    # uses to turn FluidConfig.damping_ratio into PendulumBackend's linear
    # damping_coefficient -- reused here rather than re-derived, so this
    # linearization can never silently drift from what PendulumBackend
    # actually integrates.
    c_pendulum = 2.0 * fluid_config.damping_ratio * housner.natural_frequency_rad_s

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -b_tank / inertia, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, -gravity / pendulum_length, -c_pendulum],
        ],
        dtype=np.float64,
    )
    B = np.array([[0.0], [1.0 / inertia], [0.0], [0.0]], dtype=np.float64)

    return A, B


class SloshLQRController:
    """Continuous-time infinite-horizon LQR state feedback: u = -K @ x.

    Q/R are diagonal by default -- penalizing tank angle/rate and pendulum
    angle/rate independently, with R sized against
    EnvironmentConfig.max_motor_torque_nm so the resulting gains produce
    commands on the same physical scale the environment actually clips
    to, rather than requiring after-the-fact re-tuning.
    """

    def __init__(
        self,
        tank_config: Optional[TankConfig] = None,
        pivot_config: Optional[PivotConfig] = None,
        fluid_config: Optional[FluidConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        world_config: Optional[WorldConfig] = None,
        q_diag: Sequence[float] = (10.0, 1.0, 50.0, 1.0),
        r_value: Optional[float] = None,
    ) -> None:
        resolved_tank_config = tank_config or TankConfig(fixed=False)
        resolved_fluid_config = fluid_config or FluidConfig()
        resolved_world_config = world_config or WorldConfig()

        self._env_config = env_config or EnvironmentConfig()
        self._torque_limit = self._env_config.max_motor_torque_nm
        self._housner = compute_housner_parameters(
            tank_length=resolved_tank_config.length,
            tank_breadth=resolved_tank_config.width,
            fill_height=resolved_fluid_config.fill_height,
            fluid_density=resolved_fluid_config.density,
            gravity=abs(resolved_world_config.gravity[1]),
        )

        # Cached only for get_action_from_pressure_observation()'s analytic
        # inversion of environment/observable_sloshing_env.py::wall_pressures()
        # -- kept as the project's own real config values, not re-derived
        # constants, so this can't silently drift from what that function
        # actually computes.
        self._half_length = resolved_tank_config.length / 2.0
        self._fill_height = resolved_fluid_config.fill_height
        self._fluid_density = resolved_fluid_config.density
        self._gravity = abs(resolved_world_config.gravity[1])

        self.A, self.B = build_state_space(
            tank_config, pivot_config, fluid_config, self._env_config, world_config
        )

        self.Q = np.diag(q_diag).astype(np.float64)
        # R sized to 1/torque_limit^2 by default: a control effort of
        # exactly the actuator's own limit incurs unit cost, so Q's
        # relative weights are directly comparable to "how many limit-size
        # torque pulses is this state error worth."
        r = r_value if r_value is not None else 1.0 / (self._torque_limit ** 2)
        self.R = np.array([[r]], dtype=np.float64)

        if not self.is_controllable():
            raise ValueError(
                "(A, B) pair is not controllable -- check tank_config/pivot_config/"
                "fluid_config for a degenerate combination (e.g. zero inertia)."
            )

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, self.B.T @ self.P)  # 1x4

    def is_controllable(self) -> bool:
        n = self.A.shape[0]
        controllability = np.hstack([np.linalg.matrix_power(self.A, i) @ self.B for i in range(n)])
        return np.linalg.matrix_rank(controllability) == n

    def poles(self) -> np.ndarray:
        """Closed-loop eigenvalues of (A - B@K) -- all should have negative real parts."""
        closed_loop = self.A - self.B @ self.K
        return np.linalg.eigvals(closed_loop)

    def get_action(self, state: Sequence[float]) -> float:
        """state = [theta, theta_dot, phi, phi_dot]. Returns clipped torque (N*m)."""
        x = np.asarray(state, dtype=np.float64).reshape(4, 1)
        u = float(-(self.K @ x)[0, 0])
        return max(-self._torque_limit, min(self._torque_limit, u))

    def get_action_from_observation(self, observation: Sequence[float]) -> float:
        """Convenience wrapper for environment/sloshing_env.py's real 7-D
        observation, [theta, theta_dot, com_x, com_y, com_vx, com_vy,
        slosh_energy]. phi (the pendulum's own relative angle) is not
        directly observable in that vector, so it's recovered via the
        SAME small-angle proxy main.py::_surface_tilt_rad already uses
        (fluid_com_x / pendulum_length == sin(phi) exactly, per
        PendulumBackend.get_state()'s own com_position formula) --
        not a new, independently-invented approximation.
        """
        theta, theta_dot, com_x, _com_y, com_vx, _com_vy, _slosh_energy = observation
        length = self._housner.pendulum_length
        phi = math.asin(max(-1.0, min(1.0, com_x / length)))
        cos_phi = math.cos(phi)
        phi_dot = (com_vx / (length * cos_phi)) if abs(cos_phi) > 1e-9 else 0.0
        return self.get_action([theta, theta_dot, phi, phi_dot])

    def get_action_from_pressure_observation(self, observation: Sequence[float]) -> float:
        """Convenience wrapper for environment/observable_sloshing_env.py's
        real 7-D sensor-only observation, [theta, theta_dot, pressure_left,
        pressure_right, delta_pressure, delta_pressure_dot, previous_torque].

        Unlike get_action_from_observation() above, this NEVER touches the
        privileged fluid_com the base SloshingSuppressionEnv exposes -- it
        recovers com_x/com_vx by analytically inverting
        environment/observable_sloshing_env.py::wall_pressures() instead,
        so an LQR benchmarked this way is on the exact same hardware-
        realistic footing as SloshPIDController and a PPO policy trained
        against ObservableSloshingEnv (see
        benchmarks/run_observable_controller_benchmarks.py).

        The inversion is exact, not approximate, given wall_pressures()'s
        own formula: its dynamic_head term is identical on both walls, so
        it cancels completely in delta_pressure = pressure_left -
        pressure_right, leaving delta_pressure linear in com_x alone:

            delta_pressure = -(fluid_density * gravity * fill_height / half_length) * com_x

        which is what's inverted here (same relationship differentiated,
        via delta_pressure_dot, for com_vx). phi/phi_dot are then recovered
        from com_x/com_vx exactly as get_action_from_observation() already
        does -- the SAME small-angle proxy, not a second one invented here.
        """
        theta, theta_dot, _p_left, _p_right, delta_pressure, delta_pressure_dot, _prev_torque = observation
        scale = self._fluid_density * self._gravity * self._fill_height / self._half_length
        com_x = -float(delta_pressure) / scale
        com_vx = -float(delta_pressure_dot) / scale

        length = self._housner.pendulum_length
        phi = math.asin(max(-1.0, min(1.0, com_x / length)))
        cos_phi = math.cos(phi)
        phi_dot = (com_vx / (length * cos_phi)) if abs(cos_phi) > 1e-9 else 0.0
        return self.get_action([theta, theta_dot, phi, phi_dot])


if __name__ == "__main__":
    print("=" * 60)
    print("LQR CONTROLLER SELF-TEST (linearized from this project's real config)")
    print("=" * 60)

    controller = SloshLQRController()

    print(f"\nA matrix:\n{controller.A}")
    print(f"\nB matrix:\n{controller.B}")
    print(f"\nControllable: {controller.is_controllable()}")
    print(f"\nLQR gain K: {controller.K}")

    print("\nClosed-loop poles (all should have negative real parts for stability):")
    for pole in controller.poles():
        print(f"  {pole:.4f}  (stable: {pole.real < 0})")

    print("\nSimulating a small initial disturbance under LQR feedback:")
    state = np.array([0.12, 0.0, 0.06, 0.0])
    dt = 0.02
    for step in range(50):
        u = controller.get_action(state)
        A, B = controller.A, controller.B
        state_dot = (A @ state.reshape(4, 1) + B * u).flatten()
        state = state + state_dot * dt
        if step % 10 == 0:
            print(
                f"  t={step * dt:4.2f}s  theta={math.degrees(state[0]):+6.2f} deg  "
                f"phi={math.degrees(state[2]):+6.2f} deg  u={u:+5.2f} N*m"
            )

    print("=" * 60)
