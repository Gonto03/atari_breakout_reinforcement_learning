import gymnasium
import sys
sys.modules["gym"] = gymnasium
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random

import stable_baselines3
from stable_baselines3 import DQN


env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()
print()


# ENVIRONMENT CHANGES
reward_wrapper = gymnasium.wrappers.TransformReward(env, lambda r: 2*r)
_ = env.reset()

# observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
# print(reward)


def training():
    model = DQN("CnnPolicy", env, learning_rate=0.001, buffer_size=100000, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4, progress_bar=True, reset_num_timesteps=False)
    model.save("dqn_breakout")

def testing():
    model = DQN.load("dqn_breakout")
    episodes = 5

    for _ in range(episodes):
        observation, info = env.reset()
        terminated = False
        score = 0
        observation, reward, terminated, truncated, info = env.step(1)  # start game
        n_lives = info['lives']
        
        while not terminated:
            action, _states = model.predict(observation, deterministic=True)
            # action = random.choice([0,2,3])
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            if info['lives'] == 0:
                break
            if n_lives != info['lives']:
                observation, reward, terminated, truncated, info = env.step(1)   # after losing a life, restarts the game
                n_lives = info['lives']
            env.render()
            
        print(f"Episode {_+1}\n Score: {score}")

# training()
testing()

env.close()
