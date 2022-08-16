import setup_path
import gym
import airgym
import time

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = TD3( #action should be continue
    "CnnPolicy",
    env,
    learning_rate=0.00001,
    verbose=1,
    batch_size=64,
    train_freq=4,
    learning_starts=1000,
    buffer_size=1000000,
    device="auto",
    target_policy_noise=0.5,
    target_noise_clip=1.0,
    tensorboard_log="./tb_logs/",
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
    verbose = 1
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5, tb_log_name="TD3_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("TD3_airsim_car_policy")
