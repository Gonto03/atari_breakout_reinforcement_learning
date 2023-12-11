import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random

env = gym.make("ALE/Breakout-v5", render_mode='human', full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()

episodes = 10

for _ in range(episodes):
    state = env.reset()
    terminated = False
    score = 0
    observation, reward, terminated, truncated, info = env.step(1)  # start game
    n_lives = 5
    
    while not terminated:
        action = random.choice([0,2,3])     # choose between noop, move left and move right
        observation, reward, terminated, truncated, info = env.step(action)
        # print(f"observation: {observation}")
        # print(f"reward: {reward}")
        # print(f"terminated: {terminated}")
        # print(f"truncated: {truncated}")
        # print(f"info: {info}")
        score += reward
        if info['lives'] == 0:
            break
        if n_lives != info['lives']:
            observation, reward, terminated, truncated, info = env.step(1)   # after losing a life, restarts the game
            n_lives = info['lives']
        env.render()
        
    print(f"Episode {_+1}\n Score: {score}")

env.close()
