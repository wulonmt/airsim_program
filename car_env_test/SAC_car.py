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
from CustomSAC import CustomSAC

import argparse
import json
from subprocess import Popen
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", help="which track will be used, 0~2", type=int, default=1)
parser.add_argument("-i", "--intersection", help="which intersection car is in", type=int, default=1)
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
parser.add_argument("-m", "--model", help="model path to load", type=str, nargs='?')
args = parser.parse_args()

with open("settings.json") as f:
    settings = json.load(f)
#settings["ViewMode"] = "SpringArmChase"
settings["ViewMode"] = "NoDisplay"
Car = settings["Vehicles"]["Car1"]
if args.track == 1:
    if args.intersection == 1:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, 0, 0, 180)
    elif args.intersection == 2:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (-127, 0, 0, 270)
    elif args.intersection == 3:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (-127, -128, 0, 0)
    elif args.intersection == 4:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, -128, 0, 90)
    else:
        Car["X"], Car["Y"], Car["Z"], Car["Yaw"] = (0, 0, 0, 180)
settings["Vehicles"]["Car1"] = Car
with open("settings.json", "w") as f:
    json.dump(settings, f, indent=4)

print(Popen("./Environment.sh"))
time.sleep(7) #wait for airsim opening"

# Create a DummyVecEnv for main airsim gym env
env = gym.make(
                "airgym:airsim-car-cont-action-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
env.env.setkwargs(track = args.track)
env.env.setInitialPos(Car["X"], Car["Y"], Car["Z"])
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

if args.model is not None: #load the trained model
    model = CustomSAC.load(
        args.model,
        env,
        learning_rate=0.0003,
        verbose=1,
        batch_size=64,
        train_freq=1,
        learning_starts=500, #testing origin 1000
        buffer_size=200000,
        device="auto",
        tensorboard_log="./eval_logs/",
        ent_coef = "auto_1"
    )
else:
    # Initialize RL algorithm type and parameters
    model = CustomSAC( #action should be continue
        "CnnPolicy",
        env,
        learning_rate=0.0003,
        verbose=1,
        batch_size=64,
        train_freq=1,
        learning_starts=500, #testing origin 1000
        buffer_size=200000,
        device="auto",
        tensorboard_log="./tb_logs/",
        ent_coef = "auto_1",
        target_entropy = -2.0,
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
  check_episodes=1e3,
  save_path="./checkpoint/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
  verbose=2
)
#callback_list.append(ep_checkpoint_callback)

# Stops training when the model reaches the maximum number of episodes
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e4, verbose=1)
#callback_list.append(callback_max_episodes)

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
    total_timesteps=2e4, tb_log_name=t + f"inter{args.intersection}" + f"_{args.log_name}", callback = callback
)

if args.model is None: #If model is not trained model, save it
    # Save policy weights
    if not os.path.isdir('result_model'):
        os.mkdir('result_model')
    model.save("result_model/" + t + f"inter{args.intersection}" + f"_{args.log_name}")
