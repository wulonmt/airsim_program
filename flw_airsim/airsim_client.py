import setup_path
import gym
import airgym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import flwr as fl
import numpy as np
from collections import OrderedDict
import torch as th
from torch import nn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--track", help="which track will be used, 0~2", type=int)
args = parser.parse_args()

class AirsimClient(fl.client.NumPyClient):
    def __init__(self):
        # Create a DummyVecEnv for main airsim gym env
        self.env = gym.make(
                        "airgym:airsim-car-cont-action-sample-v0",
                        ip_address="127.0.0.1",
                        image_shape=(84, 84, 1),
                    )
        self.env.env.setkwargs(track = args.track)
        self.env = DummyVecEnv(
            [
                lambda: Monitor(
                    self.env
                )
            ]
        )

        # Wrap env as VecTransposeImage to allow SB to handle frame observations
        self.env = VecTransposeImage(self.env)

        #Custum Input
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
        )

        # Initialize RL algorithm type and parameters
        self.model = SAC( #action should be continue
            "CnnPolicy",
            self.env,
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
        callbacks = []
        eval_callback = EvalCallback(
            self.env,
            callback_on_new_best=None,
            n_eval_episodes=5,
            best_model_save_path=".",
            log_path=".",
            eval_freq=5000,
            verbose = 1
        )
        callbacks.append(eval_callback)
        
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
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e3, verbose=1)
        callback_list.append(callback_max_episodes)

        self.callback = CallbackList(callback_list)

        #make time eazier to read
        Ttime = str(time.ctime())
        Ttime = Ttime.split(" ")
        Ttime.reverse()
        self.time =  ""
        for t in Ttime:
            self.time += (t + "_")
        print("Start time: ", self.time)
        self.n_round = int(0)
        
    def get_parameters(self, config):
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        return policy_state

    def set_parameters(self, parameters):
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.n_round += 1
        self.set_parameters(parameters)
        self.model.learn(total_timesteps=2e4, tb_log_name=self.time + f"/SAC_airsim_car_round_{self.n_round}", reset_num_timesteps=False, callback = self.callback)
        return self.get_parameters(config={}), self.model.buffer_size, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        return -reward_mean, self.model.buffer_size, {"reward mean": reward_mean, "reward std": reward_std} 

def main():        
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="192.168.1.85:8080",
        client=AirsimClient(),
    )
if __name__ == "__main__":
    main()


