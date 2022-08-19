import setup_path
import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-disc-action-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)
log_dir = "./tb_logs/"

# Initialize RL algorithm type and parameters
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.001,
    verbose=1,
    batch_size=32,
    clip_range=0.5,
    max_grad_norm=10,
    device="cuda",
    tensorboard_log=log_dir,
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5, tb_log_name="PPO_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("PPO_airsim_car_policy")
