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


env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')

env.reset()

#num of possible actions
num_actions = env.action_space.n    #   = 4 (length of "meaning")
#action space itself
meaning = env.unwrapped.get_action_meanings()
#[NOOP(no operation), FIRE(starts game), RIGHT(moves paddle right), LEFT(moves paddle left)]

keybinds={'n':0, 's':1, 'd':2, 'a':3}

obs, reward, terminated, truncated, info = env.step(1)  #obs=3Darray(width, height, 3(rgb)), terminated=boolean
#reward: red=7,orange=7,yellow=4,green=4,light blue=1,blue=1
#info = number of lifes, current frame

print(obs[75][100])

#shows current frame
plt.figure(figsize=(8, 8))
plt.imshow(obs)


#play the game
def testing():
    model = DQN.load("dqn_breakout")
    episodes = 5

    for _ in range(episodes):
        observation, info = env.reset()
        terminated = False
        score = 0
        observation, reward, terminated, _, info = env.step(1)  # start game
        n_lives = info['lives']
        
        while not terminated:
            action, _states = model.predict(observation, deterministic=True)
            # action = random.choice([0,2,3])
            observation, reward, terminated, _, info = env.step(action)
            score += reward
            if info['lives'] == 0:
                break
            if n_lives != info['lives']:
                observation, reward, terminated, _, info = env.step(1)   # after losing a life, restarts the game
                n_lives = info['lives']
            env.render()
            
        print(f"Episode {_+1}\n Score: {score}")

# testing()

env.reset()
env.close()