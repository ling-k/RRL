import os
import logging
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import results_plotter
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy,CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from BoxWorldEnv import BoxWorldEnvImage
from Lib.mylibw import MovingFn

logging.basicConfig(filename='boxmlp4.log',level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

best_mean_reward, n_steps = -np.inf, 0
avgSuccess = MovingFn( np.mean,100 )


ep=0
def callback(_locals, _globals):
    global n_steps, best_mean_reward,ep,avgSuccess,logging
    
    if np.sum( _locals['masks']>0) >0:
        ep+=1    
        d=0
        if ( np.sum( _locals['true_reward'] ) >.1):
            d=1
        print(ep, avgSuccess.add(d))
        logging.info("%d                                  ,%.2f"%(ep,avgSuccess.get() ))
    
    n_steps += 1
    return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

env = DummyVecEnv([lambda:  BoxWorldEnvImage(4, 20, goal_type='stack buttom blue', reward=1,penalty=0,error_penalty=0)for _ in range(1)])
model = A2C(CnnPolicy, env, verbose=0,tensorboard_log=None,  full_tensorboard_log=False, learning_rate=0.001, gamma=0.9)

time_steps = 1e8
model.learn(total_timesteps=int(time_steps), callback=callback)

