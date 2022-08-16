import setup_path
import gym
import airgym
import time
from airgym.envs.car_env import AirSimCarEnv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = AirSimCarEnv("127.0.0.1", (84,84,1))
    
    while(True):
        env._do_action([0, 0.6]) #前進
        """
        img = env._get_obs()
        plt.imshow(img)
        plt.pause(1) #這樣才會讓圖片持續1秒後關掉
        print("plt show")
        time.sleep(1)
        plt.close('all')
        print("plt close")
        
        """
        
        #print(env.car.getCarState().kinematics_estimated.position.to_numpy_array())
        env._get_obs()
        print(env.state["pose"].position.to_numpy_array())
        time.sleep(1)
        env._do_action([0, 0.5])
        time.sleep(1)
