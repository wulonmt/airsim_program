import setup_path
import gym
import airgym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import flwr as fl
import numpy as np
from collections import OrderedDict
import torch


class AirsimClient(fl.client.NumPyClient):
    def __init__(self):
        # Create a DummyVecEnv for main airsim gym env
        env = DummyVecEnv(
            [
                lambda: Monitor(
                    gym.make(
                        "airgym:airsim-car-cont-action-sample-v0",
                        ip_address="127.0.0.1",
                        image_shape=(84, 84, 1),
                    )
                )
            ]
        )

        # Wrap env as VecTransposeImage to allow SB to handle frame observations
        self.env = VecTransposeImage(env)

        # Initialize RL algorithm type and parameters
        self.model = SAC( #action should be continue
            "CnnPolicy",
            env,
            learning_rate=0.0003,
            verbose=1,
            batch_size=64,
            train_freq=1,
            learning_starts=1000,
            buffer_size=200000,
            device="auto",
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

        self.callback_kwargs = {}
        self.callback_kwargs["callback"] = callbacks
        
    def get_parameters(self, config):
        policy_state = [value.cpu().numpy() for key, value in self.model.policy.state_dict().items()]
        #print("policy_state: ", policy_state)
        print("policy_state type: ", type(policy_state))
        return policy_state

    def set_parameters(self, parameters):
        #print("parameters: ", parameters)
        print("parameters type: ", type(parameters))
        params_dict = zip(self.model.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.policy.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.learn(total_timesteps=10, **self.callback_kwargs)
        return self.get_parameters(config={}), self.model.buffer_size, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        reward_mean, reward_std = evaluate_policy(self.model, self.env)
        return 1/reward_mean, self.model.buffer_size, {"reward mean": reward_mean, "reward std": reward_std} 

def main():        
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=AirsimClient(),
    )
if __name__ == "__main__":
    main()


