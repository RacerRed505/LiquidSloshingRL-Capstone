"""Classical control baselines (PID, LQR) for the sloshing-suppression task.

Deliberately PyChrono-free, same discipline as physics/pendulum_backend.py
and environment/tank_dynamics.py -- these controllers are plain state-in,
torque-out objects, sized from this project's own real config dataclasses
(config/simulation_config.py) rather than an independently-invented tank
model, so a benchmark comparison against the real trained PPO policy
(models/ppo_sloshing*.zip) on the real environment/sloshing_env.py is an
honest apples-to-apples comparison.
"""
