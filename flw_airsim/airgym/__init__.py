from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-car-cont-action-sample-v0", entry_point="airgym.envs:AirSimCarEnvContAction",
)

register(
    id="airsim-car-disc-action-sample-v0", entry_point="airgym.envs:AirSimCarEnvDiscAction",
)
