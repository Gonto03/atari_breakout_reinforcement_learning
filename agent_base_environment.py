import gymnasium
import sys
sys.modules["gym"] = gymnasium
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from environment_customization import CustomRewardBreakout
from performance_metrics import SaveTrainingResults
#####

env = gymnasium.make("ALE/Breakout-v5", render_mode="human", full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()
# env = CustomRewardBreakout(env)   # re-mapping the rewards

def training():
    model = DQN("CnnPolicy", env, learning_rate=0.001, buffer_size=10000, verbose=1)
    model.learn(total_timesteps=1000, log_interval=10, progress_bar=True, reset_num_timesteps=False)
    model.save("dqn_breakout")

# training()

model = DQN.load("dqn_breakout")
episodes = 10

for _ in range(episodes):
    observation, info = env.reset()
    terminated = False
    score = 0
    observation, reward, terminated, truncated, info = env.step(1)  # start game
    n_lives = info['lives']
    
    while not terminated:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        if info['lives'] == 0:
            break
        if n_lives != info['lives']:
            observation, reward, terminated, truncated, info = env.step(1)   # after losing a life, restarts the game
            n_lives = info['lives']
        env.render()
        
    print(f"Episode {_+1}\n Score: {score}")

env.close()
