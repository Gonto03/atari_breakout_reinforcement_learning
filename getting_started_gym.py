import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

episodes = 10

for _ in range(episodes):
    state = env.reset()
    terminated = False
    score = 0
    
    while not terminated:
        action = random.choice([0,1])
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
        
    print(f"Episode {_}\n Score: {score}")

env.close()