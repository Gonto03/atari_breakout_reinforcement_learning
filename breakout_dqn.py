import gymnasium
import sys
sys.modules["gym"] = gymnasium
import keyboard
import time
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

env = gymnasium.make("ALE/Breakout-v5", render_mode='human', full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')

#num of possible actions
num_actions = env.action_space.n    #   = 4 (length of "meaning")
#action space itself
meaning = env.unwrapped.get_action_meanings()
#[NOOP(no operation), FIRE(starts game), RIGHT(moves paddle right), LEFT(moves paddle left)]

keybinds={'n':0, 's':1, 'd':2, 'a':3}

model = DQN("CnnPolicy", env, verbose=1, buffer_size=100000)
model.learn(total_timesteps=25000)
model.save("deepq_breakout")

del model

model = DQN.load("deepq_breakout")

obs, info = env.reset()

'''
obs, reward, terminated, truncated, info = env.step(1)  #obs=3Darray(width, height, 3(rgb)), terminated=boolean
#reward: red=7,orange=7,yellow=4,green=4,light blue=1,blue=1
#info = number of lifes, current frame
'''

'''
#shows current frame
plt.figure(figsize=(8, 8))
plt.imshow(obs)
'''

#play the game
total_reward = 0
while True:
    #event = keyboard.read_event()   #returns -1 if key is not on keybinds
    #if keybinds.get(event.name, -1) != -1:
    #    obs, reward, terminated, truncated, info = env.step(keybinds.get(event.name, -1))
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
    env.render()

env.reset()
env.close()