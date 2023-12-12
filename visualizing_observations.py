import gymnasium
import sys
sys.modules["gym"] = gymnasium
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
import h5py

import stable_baselines3
from stable_baselines3 import DQN

env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False, repeat_action_probability=0.1, obs_type='rgb')
observation, info = env.reset()

episodes = 1
model = DQN.load("dqn_breakout")

def see_observation(observation):
    for i in range(210):
        for j in range(160):
            print(f"{i},{j}: {observation[i][j]}")
    
see_observation(observation)

# for _ in range(episodes):
#     state = env.reset()
#     terminated = False
#     score = 0
#     observation, reward, terminated, truncated, info = env.step(1)  # start game
#     n_lives = info['lives']
    
#     action, _states = model.predict(observation, deterministic=True)
#     observation, reward, terminated, truncated, info = env.step(action)
#     score += reward
#     if info['lives'] == 0:
#         break
#     if n_lives != info['lives']:
#         observation, reward, terminated, truncated, info = env.step(1)   # after losing a life, restarts the game
#         n_lives = info['lives']
#     env.render()
    
#     print(f"Episode {_+1}\n Score: {score}")

env.close()
