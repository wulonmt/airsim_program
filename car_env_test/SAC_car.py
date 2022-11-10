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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "img":
                # We assume CxHxW images (channels first)
                # Re-ordering will be done by pre-preprocessing or wrapper
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                # Compute shape by doing one forward pass
                with th.no_grad():
                    ex_shape = extractors[key](th.as_tensor(observation_space.sample()[key]).float())
                    
                linear = nn.Sequential(nn.Linear(ex_shape.shape[0] * ex_shape.shape[1], 256, nn.ReLU()))  #256 is img features dim
                extractors[key] = nn.Sequential(extractors[key], linear)
                total_concat_size += 256

                #self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
                
            elif key == "sp":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)

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
env = VecTransposeImage(env)

#Custum Input
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
)

# Initialize RL algorithm type and parameters
model = SAC( #action should be continue
    "MultiInputPolicy",
    env,
    learning_rate=0.0003,
    policy_kwargs=policy_kwargs,
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
print(model.policy)
# Train for a certain number of timesteps
#model.learn(
#    total_timesteps=5e6, tb_log_name="SAC_airsim_car_run_" + str(time.time()), **kwargs
#)

# Save policy weights
#model.save("SAC_airsim_car_policy")
