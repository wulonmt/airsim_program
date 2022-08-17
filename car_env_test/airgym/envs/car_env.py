import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

import numpy as np
from numpy import savetxt

action_file = "action.csv"
class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

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
        
        high = np.array([0.9, 1]) #方向，煞車油門
        low = np.array([-0.9, 0])
        
        self.action_space = spaces.Box(low , high)

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None
        

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        if float(action[1]) > 0.5 :
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
        else:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1

        """
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
        """
        
        print("action = ", action)
        """
        with open(action_file, 'a') as f:
            savetxt(f, action, delimiter = ',')
        """
        #print("action type = ", type(action))
        self.car_controls.steering = float(action[0])
        
        
        
        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 20 #原先似乎太大
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
        
        done = 0            
        if dist > THRESH_DIST:
            reward = -2
            done = 1
            print("Done -- distance Out\n")
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            speed = self.car_state.speed
            if speed > MAX_SPEED:
                speed = MAX_SPEED
            reward_speed = (
                (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            
            #reward = reward_dist + reward_speed
            reward = reward_dist + reward_speed + 1 #因為很多reward都小於0所以+1看看
            #reward = reward_speed
            print("%-10s" % "dist rew",': %8.3f'%reward_dist, "%-6s" % "dist", ': %.3f'%dist)
            print("%-10s" % "speed rew", ': %8.3f'%reward_speed, "%-6s" % "speed", ': %.3f'%self.car_state.speed)
            print("%-10s" % "reward", ': %8.3f'%float(reward))
            print()
            if reward < -0.95:
                done = 1
                print("Done -- reward < -1\n")
                
            elif self.car_controls.brake == 0:
                if self.car_state.speed <= 1:
                    done = 1
                    print("Done -- Speedless")
            elif self.state["collision"]:
                print("Done -- collision\n")
                done = 1
        
        """
        done = 0
        if reward < -1:
            done = 1
            print("done -- reward < -1\n")
            
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
                
        if self.state["collision"]:
            print("done -- collision\n")
            reward = -3
            done = 1
        """

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action([0, 0.5])
        return self._get_obs()
