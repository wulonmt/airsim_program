import setup_path
import gym
import airgym
import time
from torch import nn
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack

from EpisodeCheckpointCallback import EpisodeCheckpointCallback

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", help="Which track will be used, 0~2", type=int)
args = parser.parse_args()

# Create a DummyVecEnv for main airsim gym env
env = gym.make(
                "airgym:airsim-car-cont-action-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
env.env.setkwargs(track = args.track)
env = DummyVecEnv(
    [
        lambda: Monitor(
            env
        )
    ]
)

# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = SAC( #action should be continue
    "CnnPolicy",
    env,
    learning_rate=0.0003,
    verbose=1,
    batch_size=64,
    train_freq=1,
    learning_starts=1000, #testing origin 1000
    buffer_size=200000,
    device="auto",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callback_list = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
    verbose = 1
)
callback_list.append(eval_callback)

# Save a checkpoint every 1000 steps
ep_checkpoint_callback = EpisodeCheckpointCallback(
  check_episodes=5,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
  verbose=2
)
callback_list.append(ep_checkpoint_callback)

# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e4, verbose=1)
callback_list.append(callback_max_episodes)

callback = CallbackList(callback_list)

#make time eazier to read
Ttime = str(time.ctime())
Ttime = Ttime.split(' ')
if '' in Ttime:
    Ttime.remove('')
if(int(Ttime[2]) < 10):
    Ttime[2] = "0" + Ttime[2]
t =  ""
mask = [4, 1, 2, 0, 3]
for i in mask:
    t += Ttime[i] + "_"
print("Start time: ", t)

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e7, tb_log_name="SAC_airsim_car_run_" + str(time.time()), callback = callback
)

# Save policy weights
model.save("SAC_airsim_car_policy")
