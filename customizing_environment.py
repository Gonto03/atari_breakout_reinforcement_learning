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

class CustomRewardBreakout(Wrapper):
    def __init__(self, env):
        super(CustomRewardBreakout, self).__init__(env)

    def step(self, n_lives, action):
        # Perform the action in the original environment
        observation, reward, done, _, info = self.env.step(action)
        if info['lives'] != n_lives:
            reward -= 10        # reward de -10 atribuída ao agente quando este perde uma vida
        if reward > 0:          # se algum bloco foi partido, verificar quais as coordenadas deste
            pass################################

        return observation, reward, done, info

# Create the Breakout environment
env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()

# Wrap the environment with your custom reward wrapper
env = CustomRewardBreakout(env)

def testing():
    model = DQN.load("dqn_breakout")
    episodes = 5

    for _ in range(episodes):
        observation, info = env.reset()
        terminated = False
        score = 0
        observation, reward, terminated, info = env.step(1)  # start game
        n_lives = info['lives']
        
        while not terminated:
            action, _states = model.predict(observation, deterministic=True)
            # action = random.choice([0,2,3])
            observation, reward, terminated, info = env.step(n_lives, action)
            score += reward
            if info['lives'] == 0:
                break
            if n_lives != info['lives']:
                observation, reward, terminated, info = env.step(1)   # after losing a life, restarts the game
                n_lives = info['lives']
            env.render()
            
        print(f"Episode {_+1}\n Score: {score}")

testing()