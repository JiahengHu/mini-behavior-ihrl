from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.grid import is_obj
from mini_behavior.actions import Pickup, Drop, Toggle, Open, Close
from mini_behavior.objects import Wall
from mini_bddl import ACTION_FUNC_MAPPING
from mini_behavior.floorplan import *

from enum import IntEnum
from gym import spaces
import math
from .thawing_frozen_food import ThawingFrozenFoodEnv

# obj_in_scene={'fish': 1}
obj_in_scene={'olive': 1, 'fish': 1, 'date': 1}
# Initialize action space basedd on objects in scene
action_list = ["move_frig", "move_sink", "open", "close"]
for key, value in obj_in_scene.items():
    # We assume each type of object only appears once
    assert value == 1
    action_list.append("move_" + key)
    action_list.append("pickup_" + key)
    action_list.append("drop_" + key)
Actions = IntEnum('Actions', action_list, start=0)

class SimpleThawingFrozenFoodEnv(ThawingFrozenFoodEnv):
    """
    Thawing
    This is a wrapper around the original mini-behavior environment where states are represented by category, and
    actions are converted to integer selection
    """
    def __init__(
            self,
            mode='not_human',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=50,
            use_stage_reward=False,
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         obj_in_scene=obj_in_scene,
                         )

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_dim = len(self.actions)

        self.reward_range = (-math.inf, math.inf)
        self.init_stage_checkpoint()

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"frig_open": False, "succeed": False}
        for key, _ in self.obj_in_scene.items():
            self.stage_checkpoints[key + "_pickup"] = False
            self.stage_checkpoints[key + "_thaw"] = False
        self.stage_completion_tracker = 0

    def update_stage_checkpoint(self):
        self.stage_completion_tracker += 1
        if not self.stage_checkpoints["frig_open"]:
            if self.frig_open:
                self.stage_checkpoints["frig_open"] = True
                return 1
        for obj_name, obj_in_hand, obj_frozen in zip(self.obj_name_list, self.obj_inhand, self.obj_frozen):
            if not self.stage_checkpoints[obj_name + "_pickup"]:
                if obj_in_hand:
                    self.stage_checkpoints[obj_name + "_pickup"] = True
                    return 1
            if not self.stage_checkpoints[obj_name + "_thaw"]:
                if obj_frozen == 0:
                    self.stage_checkpoints[obj_name + "_thaw"] = True
                    return 1
        if not self.stage_checkpoints["succeed"]:
            if self._end_conditions():
                self.stage_checkpoints["succeed"] = True
                return 1
        self.stage_completion_tracker -= 1
        return 0

    def reset(self):
        obs = super().reset()
        self.init_stage_checkpoint()
        self.adjusted_sink_pos = np.array(self.sink.cur_pos)
        return obs

    def observation_dims(self):
        obs_dims = {
            "agent_pos": np.array([self.room_size, self.room_size]),
            "agent_dir": np.array([4]),
            "sink_pos": np.array([self.room_size, self.room_size]),
            "frig_pos": np.array([self.room_size, self.room_size]),
            "frig_state": np.array([2]),
            "step_count": np.array([1])
        }

        for i in range(len(self.obj_name_list)):
            obs_dims[self.obj_name_list[i] + "_pos"] = np.array([self.room_size, self.room_size])
            obs_dims[self.obj_name_list[i] + "_state"] = np.array([6])

        return obs_dims

    def hand_crafted_policy(self):
        """
        A hand-crafted function to select action for next step
        Navigation is accurate
        """
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Open the frig
        if not self.frig_open and Open(self).can(self.electric_refrigerator):
            action = self.actions.open
        # If any one of the object is in frig, we go to the frig and pick it up
        elif sum(self.obj_inside) > 0:
            for obj, inside in zip(self.obj_list, self.obj_inside):
                if inside:
                    obj_name = obj.type
                    if Pickup(self).can(obj):
                        action = self.actions["pickup_"+obj_name]
                    elif not self.frig_open:
                        action = self.actions.move_frig
                    else:
                        action = self.actions["move_"+obj_name]
                    break
        elif self.sink in fwd_cell[0]:  # refrig should be in all three dimensions, sink is just in the first dimension
            if sum(self.obj_inhand) > 0:
                for obj, inhand in zip(self.obj_list, self.obj_inhand):
                    if inhand:
                        obj_name = obj.type
                        if Drop(self).can(obj):
                            action = self.actions["drop_" + obj_name]
                        else:
                            action = self.actions.move_sink
                        break
            else:
                # We're done, navigate randomly
                action = self.actions.move_frig
        else:
            action = self.actions.move_sink

        return action

    def gen_obs(self):
        self.obj_list = []
        self.obj_name_list = []
        for key, _ in self.obj_in_scene.items():
            self.obj_list.append(self.objs[key][0])
            self.obj_name_list.append(key)

        self.electric_refrigerator = self.objs['electric_refrigerator'][0]
        self.sink = self.objs['sink'][0]
        self.frig_open = int(self.electric_refrigerator.check_abs_state(self, 'openable'))

        self.obj_frozen = []
        self.obj_inhand = []
        self.obj_inside = []
        for obj in self.obj_list:
            self.obj_frozen.append(int(obj.check_abs_state(self, 'freezable')))
            self.obj_inhand.append(int(obj.check_abs_state(self, 'inhandofrobot')))
            self.obj_inside.append(int(obj.check_rel_state(self, self.electric_refrigerator, 'inside')))

        obs = {
            "agent_pos": np.array(self.agent_pos),
            "agent_dir": np.array([self.agent_dir]),
            "sink_pos": np.array(self.sink.cur_pos),
            "frig_pos": np.array(self.electric_refrigerator.cur_pos),
            "frig_state": np.array([self.frig_open]),
            "step_count": np.array([float(self.step_count) / self.max_steps])
        }

        for i in range(len(self.obj_name_list)):
            obs[self.obj_name_list[i] + "_pos"] =  np.array(self.obj_list[i].cur_pos)
            obs[self.obj_name_list[i] + "_state"] = np.array([self.obj_frozen[i]])

        return obs

    def step(self, action, evaluate_mask=True):
        # cur_fish_frozen = self.fish_frozen

        self.update_states()
        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        move_success = pickup_done = frig_manipulated = False

        if action == self.actions.move_frig:
            move_success = self.set_agent_to_neighbor(self.electric_refrigerator)
        elif action == self.actions.move_sink:
            move_success = self.set_agent_to_neighbor(self.sink)
        elif action == self.actions.open:
            if Open(self).can(self.electric_refrigerator):
                Open(self).do(self.electric_refrigerator)
                frig_manipulated = True
        elif action == self.actions.close:
            if Close(self).can(self.electric_refrigerator):
                Close(self).do(self.electric_refrigerator)
                frig_manipulated = True
        else:
            obj_action = self.actions(action).name.split('_')  # list: [action, obj]

            # try to perform action
            idx = self.obj_name_list.index(obj_action[1])
            obj = self.obj_list[idx]
            action_name = obj_action[0]
            if action_name == "pickup":
                if Pickup(self).can(obj):
                    Pickup(self).do(obj)
                    pickup_done = True
            elif action_name == "move":
                move_success = self.set_agent_to_neighbor(obj)
            elif action_name == "drop":
                self.drop_rand_dim(obj)
            else:
                raise NotImplementedError

        reward = self._reward()
        # done = self._end_conditions() or self.step_count >= self.max_steps
        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        # Todo: fix mask
        if False:
        # if evaluate_mask:
            feature_dim = 11
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)


            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            fish_pos_idxes = slice(3, 5)
            fish_state_idx = 5
            sink_pos_idxes = slice(6, 8)
            frig_pos_idxes = slice(8, 10)
            frig_state_idx = 10
            action_idx = 11

            def extract_obj_pos_idxes(obj_):
                if obj == self.fish:
                    return fish_pos_idxes
                elif obj == self.sink:
                    return sink_pos_idxes
                elif obj == self.electric_refrigerator:
                    return frig_pos_idxes
                else:
                    return None

            if action in [self.actions.move_sink, self.actions.move_frig, self.actions.move_fish]:
                if move_success:
                    mask[agent_dir_idx, action_idx] = True
                    mask[agent_pos_idxes, action_idx] = True

            if pickup_done:
                obj_pos_idxes = extract_obj_pos_idxes(obj)
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True



            if frig_manipulated:
                mask[frig_state_idx, action_idx] = True
                mask[frig_state_idx, agent_pos_idxes] = True
                mask[frig_state_idx, agent_dir_idx] = True
                mask[frig_state_idx, frig_pos_idxes] = True

            if action in [self.actions.drop_fish]:
                obj_pos_idxes = extract_obj_pos_idxes(obj)
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                if not obj.check_abs_state(self, 'inhandofrobot'):
                    mask[obj_pos_idxes, action_idx] = True

            # update freeze mask
            for cur_obj_frozen, next_obj_frozen, obj_pos_idxes, obj_state_idx in \
                [
                 [cur_fish_frozen, self.fish_frozen, fish_pos_idxes, fish_state_idx]
                ]:

                if next_obj_frozen > cur_obj_frozen:
                    mask[obj_state_idx, obj_pos_idxes] = True
                    mask[obj_state_idx, frig_pos_idxes] = True
                elif next_obj_frozen < cur_obj_frozen:
                    mask[obj_state_idx, obj_pos_idxes] = True
                    mask[obj_state_idx, sink_pos_idxes] = True
            info["local_causality"] = mask

        return obs, reward, done, info


register(
    id='MiniGrid-thawing-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv'
)
