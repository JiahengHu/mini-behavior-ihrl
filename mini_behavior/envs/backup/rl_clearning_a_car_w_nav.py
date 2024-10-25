from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.grid import is_obj
from mini_behavior.actions import Pickup, Drop, Toggle, Open, Close
from mini_behavior.objects import Wall
from mini_bddl import ACTION_FUNC_MAPPING
from mini_behavior.floorplan import *

from enum import IntEnum
from gymnasium import spaces
import math
from .cleaning_a_car import CleaningACarEnv


class SimpleCleaningACarEnv(CleaningACarEnv):
    """
    Environment in which the agent is instructed to clean a car
    This is a wrapper around the original mini-behavior environment where:
    - states are represented by category, and
    - actions are converted to integer selection
    """
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        pickup_soap = 3
        drop_soap = 4
        pickup_rag = 5
        drop_rag = 6 # Do we differentiate between drop and drop-in?
        toggle_sink = 7


    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=300,
            use_stage_reward=False
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward


        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        # We redefine action space here
        self.actions = SimpleCleaningACarEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_dim = len(self.actions)

        self.reward_range = (-math.inf, math.inf)
        self.init_stage_checkpoint()

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"rag_soaked": False, "car_not_stain": False, "succeed": False}
        self.stage_completion_tracker = 0

    def reset(self):
        obs = super().reset()
        self.init_stage_checkpoint()
        return obs

    def observation_dims(self):
        return {
            "agent_pos": np.array([self.room_size, self.room_size]),
            "agent_dir": np.array([4]),
            "car_pos": np.array([self.room_size, self.room_size]),
            "car_state": np.array([2]),
            "bucket_pos": np.array([self.room_size, self.room_size]),
            "soap_pos": np.array([self.room_size, self.room_size]),
            "sink_pos": np.array([self.room_size, self.room_size]),
            "sink_state": np.array([2]),
            "rag_pos": np.array([self.room_size, self.room_size]),
            "rag_state": np.array([6, 6]),
            "step_count": np.array([1])
        }

    def hand_crafted_policy(self):
        """
        A hand-crafted function to select action for next step
        Navigation is accurate
        """
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)


        if self.car_stain:
            if self.rag.check_abs_state(self, 'inhandofrobot'):
                if self.rag_soak != 5:
                    action = self.go_drop(self.sink, fwd_cell, 0, self.actions.drop_rag)
                else:
                    action = self.go_drop(self.car, fwd_cell, 0, self.actions.drop_rag)
            elif self.rag.check_rel_state(self, self.sink, "atsamelocation"):
                if not self.sink_toggled:
                    action = self.go_toggle(self.sink, self.actions.toggle_sink)
                elif self.rag_soak != 5:
                    action = self.sample_nav_action()
                else:
                    action = self.go_pickup(self.rag, self.actions.pickup_rag)
            else:
                action = self.go_pickup(self.rag, self.actions.pickup_rag)
        else:
            rag_in_bucket = self.rag.check_rel_state(self, self.bucket, "atsamelocation")
            soap_in_bucket = self.soap.check_rel_state(self, self.bucket, "atsamelocation")
            if rag_in_bucket and soap_in_bucket:
                action = self.sample_nav_action()
            elif not soap_in_bucket:
                if self.soap.check_abs_state(self, 'inhandofrobot'):
                    action = self.go_drop(self.bucket, fwd_cell, 0, self.actions.drop_soap)
                else:
                    action = self.go_pickup(self.soap, self.actions.pickup_soap)
            elif not rag_in_bucket:
                if self.rag.check_abs_state(self, 'inhandofrobot'):
                    action = self.go_drop(self.bucket, fwd_cell, 0, self.actions.drop_rag)
                else:
                    action = self.go_pickup(self.rag, self.actions.pickup_rag)
            else:
                print("reaching here is impossible")
                raise NotImplementedError

        return action

    def update_stage_checkpoint(self):
        self.stage_completion_tracker += 1
        if not self.stage_checkpoints["rag_soaked"]:
            if self.rag_soak == 5:
                self.stage_checkpoints["rag_soaked"] = True
                return 1
        if not self.stage_checkpoints["car_not_stain"]:
            if not self.car_stain:
                self.stage_checkpoints["car_not_stain"] = True
                return 1
        if not self.stage_checkpoints["succeed"]:
            if self._end_conditions():
                self.stage_checkpoints["succeed"] = True
                return 1
        self.stage_completion_tracker -= 1
        return 0

    def gen_obs(self):

        self.car = self.objs['car'][0]
        self.rag = self.objs['rag'][0]
        self.shelf = self.objs['shelf'][0]
        self.soap = self.objs['soap'][0]
        self.bucket = self.objs['bucket'][0]
        self.sink = self.objs['sink'][0]
        self.car_stain = int(self.car.check_abs_state(self, 'stainable'))
        self.rag_soak = int(self.rag.check_abs_state(self, 'soakable'))
        self.rag_cleanness = int(self.rag.check_abs_state(self, 'cleanness'))
        self.sink_toggled = int(self.sink.check_abs_state(self, 'toggleable'))


        obs = {
            "agent_pos": np.array(self.agent_pos),
            "agent_dir": np.array([self.agent_dir]),
            "car_pos": np.array(self.car.cur_pos),
            "car_state": np.array([self.car_stain]),
            "bucket_pos": np.array(self.bucket.cur_pos),
            "soap_pos": np.array(self.soap.cur_pos),
            "sink_pos": np.array(self.sink.cur_pos),
            "sink_state": np.array([self.sink_toggled]),
            "rag_pos": np.array(self.rag.cur_pos),
            "rag_state": np.array([self.rag_soak, self.rag_cleanness]),
            "step_count": np.array([float(self.step_count) / self.max_steps])
        }

        return obs

    def step(self, action):
        self.update_states()
        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            can_overlap = True
            for dim in fwd_cell:
                for obj in dim:
                    if is_obj(obj) and not obj.can_overlap:
                        can_overlap = False
                        break
            if can_overlap:
                self.agent_pos = fwd_pos
        elif action == self.actions.pickup_rag:
            if Pickup(self).can(self.rag):
                Pickup(self).do(self.rag)
        elif action == self.actions.pickup_soap:
            if Pickup(self).can(self.soap):
                Pickup(self).do(self.soap)
        elif action == self.actions.toggle_sink:
            if Toggle(self).can(self.sink):
                Toggle(self).do(self.sink)
        elif action == self.actions.drop_rag:
            self.drop_rand_dim(self.rag)
        elif action == self.actions.drop_soap:
            self.drop_rand_dim(self.soap)
        else:
            print(action)
            raise NotImplementedError

        reward = self._reward()
        # done = self._end_conditions() or self.step_count >= self.max_steps
        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        return obs, reward, done, info


register(
    id='MiniGrid-clearning_car-v0',
    entry_point='mini_behavior.envs:SimpleCleaningACarEnv'
)
