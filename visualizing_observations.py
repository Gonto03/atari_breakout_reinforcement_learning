import gymnasium
import sys
sys.modules["gym"] = gymnasium
import matplotlib.pyplot as plt
import numpy as np

import stable_baselines3
from stable_baselines3 import DQN

env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False, repeat_action_probability=0.1, obs_type='rgb')
observation, info = env.reset()    # retrieving the initial state's information


def see_observation(observation):   # visualizing "observation": the screen's pixel map containing the RGB code for each pixel
    for i in range(210):            # screen's size: (210,160)
        for j in range(160):
            obs = observation[i][j]   # obtaining a pixel's RGB code
            is_black = [True if obs[i]==0 else False for i in range(3)]     # [0,0,0] - black; game's empty space color
            is_grey = [True if obs[i]==142 else False for i in range(3)]    # [142,142,142] - grey; screen's borders (unplayable)
            if not all(is_black) and not all(is_grey):    # checks if the pixels aren't black nor grey
                print(f"({i},{j}): {observation[i][j]}")  # if not, we visualize this pixel's RGB code
            
see_observation(observation)

'''
OBSERVAÇÕES:
Após executarmos a função see_observation(), depreendemos o seguinte:
- os blocos azuis estão entre os píxeis (87,8) e (92,151) e o seu código RGB é (66,72,200);
- os blocos verdes estão entre os píxeis (81,8) e (86,151) e o seu código RGB é (72,160,72);
- os blocos amarelos estão entre os píxeis (75,8) e (80,151) e o seu código RGB é (162,162,42);
- a camada inferior de blocos cor de laranja estão entre os píxeis (69,8) e (74,151) e o seu código RGB é (180,122,48);
- a camada superior de blocos cor de laranja estão entre os píxeis (63,8) e (68,151) e o seu código RGB é (198,108,58);
- os blocos vermelhos estão entre os píxeis (57,8) e (62,151) e o seu código RGB é (200,72,72).
'''


env.close()
