from gym_minigrid.envs.empty import EmptyEnv
from mini_behavior.register import register
import numpy as np

class RLEmptyEnv(EmptyEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	Obj obs, added API
	"""

	def __init__(self, size=10, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
		super().__init__(
			size, agent_start_pos, agent_start_dir, **kwargs
		)
		self.action_dim = len(self.actions)
		self.room_size = size

	# observation only needs pos and dir
	# Since the goal is always fixed
	def gen_obs(self):
		obs_dict = {
			"agent_pos": np.array(self.agent_pos),
			"agent_dir": np.array([self.agent_dir]),
		}
		return obs_dict

	def observation_dims(self):
		return {
			"agent_pos": np.array([self.room_size, self.room_size]),
			"agent_dir": np.array([4]),
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
