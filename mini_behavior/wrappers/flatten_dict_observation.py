import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

class FlattenDictObservation(FlattenObservation):
    def __init__(self, env: gym.Env):
        """Flattens the observations of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        self.dict_obs_space = env.observation_space
        self.num_factors = len(env.observation_space.spaces)
        self.goal_based = False

        self.breakpoints = [0]
        self.factor_spaces = []
        for obs_k, obs_space in env.observation_space.spaces.items():
            assert isinstance(obs_space, spaces.MultiDiscrete)
            self.breakpoints.append(self.breakpoints[-1] + np.sum(obs_space.nvec))
            self.factor_spaces.append(obs_space)
        self.breakpoints = np.array(self.breakpoints)
        super().__init__(env)
