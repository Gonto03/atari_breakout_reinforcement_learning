import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class CustomMonitor(Monitor):     # monitor created by us that allows for training curve visualization
    def __init__(self, env, log_dir, **kwargs):
        self.log_dir = log_dir
        super(CustomMonitor, self).__init__(env, log_dir, **kwargs)

    def close(self):
        super(CustomMonitor, self).close()
        self.plot_results_()

    def plot_results_(self):
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')

        if len(x) > 0:
            plt.plot(x, y, label='DQN')
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.title('Training Curve')
            plt.legend()
            plt.show()




class SaveTrainingResults(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

        return True