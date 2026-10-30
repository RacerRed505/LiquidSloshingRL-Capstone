"""Phase 8 -- Layer 3, the Gymnasium environment for this project."""

from gymnasium.envs.registration import register

from environment.observable_sloshing_env import ObservableSloshingEnv
from environment.sloshing_env import SloshingSuppressionEnv
from environment.tank_dynamics import TankDynamicsModel, build_tank_dynamics_model

__all__ = [
    "ObservableSloshingEnv",
    "SloshingSuppressionEnv",
    "TankDynamicsModel",
    "build_tank_dynamics_model",
]

register(
    id="LiquidSloshingRL/SloshingSuppression-v0",
    entry_point="environment.sloshing_env:SloshingSuppressionEnv",
)

# Added 2026-09-16 alongside environment/observable_sloshing_env.py -- same
# registration pattern as the base env, so training/train_ppo_observable.py
# can use stable-baselines3's make_vec_env() exactly like training/train_ppo.py
# does, rather than a bespoke env-construction path.
register(
    id="LiquidSloshingRL/ObservableSloshingSuppression-v0",
    entry_point="environment.observable_sloshing_env:ObservableSloshingEnv",
)
