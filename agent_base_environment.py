import gymnasium
import sys
sys.modules["gym"] = gymnasium
import os
import matplotlib.pyplot as plt
import numpy as np

import stable_baselines3
from stable_baselines3 import DQN

from performance_metrics import CustomMonitor


# Create logs directory
log_dir = "logs_base/"

# creating environment
env = gymnasium.make("ALE/Breakout-v5", render_mode="human", full_action_space=False,
               repeat_action_probability=0.1,obs_type='rgb')
observation, info = env.reset()
env = CustomMonitor(env, log_dir)   # wrapping the environment with our custom monitor

def training():
    model = DQN("CnnPolicy", env, learning_rate=0.001, buffer_size=10000, verbose=1)    # creating the DQN model
    model.learn(total_timesteps=300000, log_interval=10, progress_bar=True, reset_num_timesteps=False)   # training the model
    model.save("../dqn_base_breakout")    # saving the model


def testing():      # testing the trained model
    model = DQN.load("../dqn_base_breakout")
    episodes = 10
    episode_scores = []

    for _ in range(episodes):       # the game is fully played 10 times
        observation, info = env.reset()     # reseting the environment before each episode
        terminated = False
        score = 0
        observation, reward, terminated, truncated, info = env.step(action=1)  # start the game
        n_lives = info['lives']
        
        while not terminated:      # this cycle is executed until the agent loses all its lives or the game is completed
            action, _states = model.predict(observation, deterministic=True)    # predicting the next action based on the trained model
            observation, reward, terminated, truncated, info = env.step(action)  # executing the action and retrieving the subsequent game's information
            score += reward     # the game's score is given by the reward at each step
            if info['lives'] == 0:
                break
            if n_lives != info['lives']:
                observation, reward, terminated, truncated, info = env.step(action=1)   # after losing a life, restarts the game
                n_lives = info['lives']
            env.render()
            
        episode_scores.append(score)    # saving the total score of each episode
        print(f"Episode {_+1}\n Score: {score}")
        
    episode_idx = [i for i in range(1,11)]
    plt.plot(episode_idx, episode_scores, label='DQN')  # plotting the score for each episode
    plt.xlabel('Episodes')
    plt.ylabel('Total score')
    plt.title('Testing: total score by episode')
    plt.legend()
    plt.show()

    env.close()
    
    
def main():
    training()
    env.plot_results_()
    testing()
    
main()