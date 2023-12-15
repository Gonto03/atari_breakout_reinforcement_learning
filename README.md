Introdução aos Sistemas Inteligentes e Autónomos - Projeto 2
Gonçalo Dias e Vicente Bandeira

Our project: Using a Stable Baselines reinforcement learning algorithm to train an agent to master the Atari game Breakout in its original gymnasium environment and with some modifications implemented by us, as well.

File structure:
agent_base_environment.py - The agent learns to play the game in its original environment.
agent_custom_environment.py - The agent learns to play the game in the environment with our modifications.
performance_metrics.py - Contains auxiliar classes and functions to save and plot the training results.
environment_customization.py - Contains what we used to modify the original environment.
visualizing_observations.py - Auxiliar script that allowed us to visualize the screen's pixel map in order to customize the environment's rewards.