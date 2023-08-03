import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.car_env_cont_action import AirSimCarEnvContAction

import numpy as np
from numpy import savetxt

"""
X=0,Y=0,Z=0,Yaw=0
"""
track0 = [
                (0, -1), (128, -1), (128, -128), (0, -128),
                (0, -1),
            ]
bound0 = [
                (-2, 1), (130, 1), (130, -130), (-2, -130),
                (-2, 1),
            ]
            
"""
X=0,Y=0,Z=0,Yaw=180
"""
track1 = [
                (0, -1), (-127, -1), (-127, -128), (0, -128),
                (0, -1),
            ]
bound1 = [
                (2, 1), (-129, 1), (-129, -130), (2, -130),
                (2, 1),
            ]
"""
X=0,Y=0,Z=0,Yaw=0
"""
track2 = [
                (0, -1), (-128, -1), (-128, -128), (0, -128),
                (0, -1),
            ]
bound2 = [
                (-2, 1), (130, 1), (130, -130), (-2, -130),
                (-2, 1),
            ]
            
class AirSimCarEnvDiscAction(AirSimCarEnvContAction):
    def __init__(self, ip_address, image_shape):
        super().__init__(ip_address, image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.car.confirmConnection()
        self.car.enableApiControl(True) #這裡是多加的 為了能用API
        self.action_space = spaces.Discrete(6)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )
        #CameraID, Data type, pixels as float or not, compressed or not

        self.car_controls = airsim.CarControls()
        self.car_state = None
        
        self.static_count = 0
        
        self.track = [(track0, bound0), (track1, bound1), (track2, bound2)]
        self.X_ = 0
        self.Y_ = 0
        self.Z_ = 0
        
        DamnAnimals = [] #Forgive me for cursing the animals which always breaks my training
        print("Animals: ")
        for objects in self.car.simListSceneObjects():
            if "Raccoon" in objects or "Deer" in objects:
                print(objects)
                DamnAnimals.append(objects)
                
        for animals in DamnAnimals:
            self.car.simDestroyObject(animals)
            time.sleep(0.05)
        print("Animals Cleaned Over.")


    def __del__(self):
        super().__del__()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def _get_obs(self):
        return super()._get_obs()
    """
    def _compute_reward(self):
        MAX_SPEED = 20 #原先的最大值似乎太大
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > THRESH_DIST:
            reward = -2
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            speed = self.car_state.speed
            if speed > MAX_SPEED:
                speed = MAX_SPEED
            reward_speed = (
                (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed + 1 #因為很多reward都小於0所以+1看看
            #reward = reward_speed
            
        print("dist = ", dist, " speed = ", self.car_state.speed, " reward = ", reward)
        print()

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done
    """
        
    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
        
