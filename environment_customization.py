import numpy as np
import gymnasium
from gymnasium import Wrapper
import sys
sys.modules["gym"] = gymnasium
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from tqdm import tqdm
import random

import stable_baselines3
from stable_baselines3 import DQN


def get_broken_block_coordinates(prev_obs, cur_obs):
    for i in range(57,93):
        for j in range(8,152):
            aux = [True if prev_obs[i][j][l]==cur_obs[i][j][l] else False for l in range(3)]
            if not all(aux):    # checks if a pixel has the same color as in the previous observation
                return (i,j)
                
def get_block_color(rgb):
    match rgb[0]:
        case 66:
            return 'blue'
        case 72:
            return 'green'
        case 162:
            return 'yellow'
        case 180:
            return 'low_orange'
        case 198:
            return 'high_orange'
        case 200:
            return 'red'
        case _:
            raise ValueError(f"Invalid color code: {rgb}")
    

class CustomRewardBreakout(Wrapper):
    def __init__(self, env):
        super(CustomRewardBreakout, self).__init__(env)

    def step(self, prev_obs, n_lives, action):
        rewards = {'blue': 1, 'green': 3, 'yellow': 5, 'low_orange': 7, 'high_orange': 9, 'red': 11}   # color-reward map
        # Perform the action in the original environment
        observation, reward, done, _, info = self.env.step(action)
        if info['lives'] != n_lives:
            reward -= 12        # a -12 reward is assigned to the agent when a life is lost
        if reward > 0:          # if a block is destroyed, its coordinates will be retrieved
            i,j = get_broken_block_coordinates(prev_obs, observation)   # obtaining the broken block's coordinates
            color = get_block_color(prev_obs[i][j])      # obtaining the broken block's color
            reward = rewards[color]       # assigning the modified reward to the broken block based on its color
        return observation, reward, done, info