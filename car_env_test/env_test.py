import setup_path
import gym
import airgym
import time
from airgym.envs.car_env_cont_action import AirSimCarEnvContAction
import matplotlib.pyplot as plt
import math

def Qua2Rad(x):
    return 2*math.acos(x.w_val)

def Rad2Deg(r):
    return r*180/math.pi

def print_Rad(env):
    ori = env.car.getCarState().kinematics_estimated.orientation
    #print(ori)
    print("angle: ", Rad2Deg(Qua2Rad(ori)))


if __name__ == "__main__":
    env = AirSimCarEnvContAction("127.0.0.1", (84,84,1))
    path = [[0, 0.5], [0, 0.6], [0,0.5], [0,0.6]]
    
    while(True):
        for p in path:
            env._do_action(p)
            time.sleep(1)
            #print(env.car.getCarState().kinematics_estimated.position.to_numpy_array())
            env._get_obs()
            print("bound dist: ", env.bound_dist())
            print(env.state["pose"].position.to_numpy_array())
            print_Rad(env)
            
        
        """
        img = env._get_obs()
        plt.imshow(img)
        plt.pause(1) #這樣才會讓圖片持續1秒後關掉
        print("plt show")
        time.sleep(1)
        plt.close('all')
        print("plt close")
        
        """
        

