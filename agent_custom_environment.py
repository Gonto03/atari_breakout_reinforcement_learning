import gymnasium
import sys
sys.modules["gym"] = gymnasium
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import random

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from environment_customization import CustomRewardBreakout
from performance_metrics import CustomMonitor


# Create logs directory
log_dir = "logs_custom/"

# creating environment
env = gymnasium.make("ALE/Breakout-v5", render_mode="human", full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()
env = CustomMonitor(env, log_dir)
env = CustomRewardBreakout(env)   # re-mapping the rewards

def training():
    model = DQN("CnnPolicy", env, learning_rate=0.001, buffer_size=10000, verbose=1)
    model.learn(total_timesteps=500, log_interval=10, progress_bar=True, reset_num_timesteps=False)
    model.save("../dqn_custom_breakout")


def testing():
    model = DQN.load("../dqn_custom_breakout")
    episodes = 10
    episode_scores = []

    for _ in range(episodes):
        observation, info = env.reset()
        terminated = False
        score = 0
        observation, reward, terminated, info = env._step(observation,n_lives=5,action=1)  # start game
        n_lives = info['lives']
        
        while not terminated:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, info = env._step(observation,n_lives,action)
            score += reward
            if info['lives'] == 0:
                break
            if n_lives != info['lives']:
                observation, reward, terminated, info = env._step(observation,n_lives=5,action=1)   # after losing a life, restarts the game
                n_lives = info['lives']
            env.render()
            
        episode_scores.append(score)
        print(f"Episode {_+1}\n Score: {score}")
        
    episode_idx = [i for i in range(1,11)]
    plt.plot(episode_idx, episode_scores, label='DQN')
    plt.xlabel('Episodes')
    plt.ylabel('Total score')
    plt.title('Training Curve')
    plt.legend()
    plt.show()

    env.close()


def main():
    training()
    env.plot_results()
    testing()
    
main()