from minigrid.envs.empty import EmptyEnv
from mini_behavior.register import register
import numpy as np

class RLEmptyEnv(EmptyEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	Obj obs, added API
	"""

	def __init__(self, size=10, max_steps=50, **kwargs):
		super().__init__(
			size, **kwargs
		)
		self.max_steps = max_steps
		self.action_dim = len(self.actions)
		self.room_size = size
		self.succeed = False

	def reset(self):
		obs = super().reset()
		self.succeed = False
		return obs

	# observation only needs pos and dir
	# Since the goal is always fixed
	def gen_obs(self):
		obs_dict = {
			"agent_pos": np.array(self.agent_pos),
			"agent_dir": np.array([self.agent_dir]),
			"step_count": np.array([float(self.step_count) / self.max_steps])
		}
		return obs_dict

	def step(self, action):
		obs, reward, done, info = super().step(action)
		if self.succeed:
			reward = 1
			done = self.step_count == self.max_steps
			info["success"] = 1
			info["stage_completion"] = 1
		else:
			if done and self.step_count < self.max_steps:
				info["success"] = 1
				info["stage_completion"] = 1
				self.succeed = True
				reward = 1
				done = False  # We don't terminate early
			else:
				info["success"] = 0
				info["stage_completion"] = 0
		return obs, reward, done, info

	def observation_dims(self):
		return {
			"agent_pos": np.array([self.room_size, self.room_size]),
			"agent_dir": np.array([4]),
			"step_count": np.array([1])
		}

	def observation_spec(self):
		"""
		dict, {obs_key: obs_range}
		"""
		return self.observation_dims()


register(
	id='MiniGrid-Empty-v0',
	entry_point='mini_behavior.envs:RLEmptyEnv'
)
