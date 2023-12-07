import gymnasium as gym
import time
import matplotlib.pyplot as plt

env = gym.make("ALE/Breakout-v5", render_mode = "human", full_action_space=False, repeat_action_probability=0.1, obs_type="rgb")