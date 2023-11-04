from enum import IntEnum

import math
from copy import deepcopy

import numpy as np
from gymnasium import spaces

from mini_behavior.actions import Pickup, Drop, Open, Close
from mini_behavior.floorplan import *
from mini_behavior.grid import is_obj
from .thawing_frozen_food import ThawingFrozenFoodEnv


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
            max_steps=300,
            use_stage_reward=False,
            seed=42,
            evaluate_graph=False,
            random_obj_pose=True,
            discrete_obs=True,
            obj_in_scene={'olive': 1, 'fish': 1, 'date': 1},
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward
        self.evaluate_graph = evaluate_graph
        self.discrete_obs = discrete_obs

        self.reward_range = (-math.inf, math.inf)
        self.random_obj_pose = random_obj_pose
        self.obj_name_list = list(obj_in_scene.keys())

        self.original_observation_space = spaces.Dict([
                ("agent", spaces.MultiDiscrete([self.room_size, self.room_size, 4])),
                ("frig", spaces.MultiDiscrete([self.room_size, self.room_size, 2])),
                ("sink", spaces.MultiDiscrete([self.room_size, self.room_size])),
            ])
        for obj_name in self.obj_name_list:
            self.original_observation_space[obj_name] = spaces.MultiDiscrete([self.room_size, self.room_size, 6])

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         seed=seed,
                         obj_in_scene=obj_in_scene,
                         )

        action_list = ["left", "right", "forward", "open", "close", "pickup"]
        for key, value in obj_in_scene.items():
            # We assume each type of object only appears once
            assert value == 1
            action_list.append("drop_" + key)
        Actions = IntEnum('Actions', action_list, start=0)
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

        if self.discrete_obs:
            self.observation_space = self.original_observation_space
        else:
            self.observation_space = spaces.Dict([
                ("agent", spaces.Box(low=-1, high=1, shape=[3])),
                ("frig", spaces.Box(low=-1, high=1, shape=[3])),
                ("sink", spaces.Box(low=-1, high=1, shape=[2])),
            ])
            for obj_name in self.obj_name_list:
                self.observation_space[obj_name] = spaces.Box(low=-1, high=1, shape=[3])

        self.init_stage_checkpoint()
        self.desired_goal = None

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"frig_open": False, "succeed": False}
        for obj_name in self.obj_name_list:
            self.stage_checkpoints[obj_name + "_pickup"] = False
            self.stage_checkpoints[obj_name + "_thaw"] = False
        self.stage_completion_tracker = 0

    def _gen_objs(self):
        if self.random_obj_pose:
            return super()._gen_objs()
        else:
            electric_refrigerator = self.objs['electric_refrigerator'][0]
            sink = self.objs['sink'][0]

            frig_top = (3, 3)
            self.put_obj(electric_refrigerator, *frig_top, 0)

            sink_pos = (6, 6)
            self.put_obj(sink, *sink_pos, 0)

            fridge_pos_idx = 0
            for obj_name in self.obj_name_list:
                obj = self.objs[obj_name][0]
                self.put_obj(obj, *electric_refrigerator.all_pos[fridge_pos_idx], 1)
                obj.states['inside'].set_value(electric_refrigerator, True)
                fridge_pos_idx += 1

    def place_agent(self):
        if self.random_obj_pose:
            return super().place_agent()
        else:
            self.agent_pos = np.array([3, 6])
            self.agent_dir = 0
            return self.agent_pos

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.init_stage_checkpoint()
        return obs, info

    def update_stage_checkpoint(self):
        self.stage_completion_tracker += 1
        if not self.stage_checkpoints["frig_open"]:
            if self.frig_open:
                self.stage_checkpoints["frig_open"] = True
                return 1
        for obj_name in self.obj_name_list:
            if not self.stage_checkpoints[obj_name + "_pickup"]:
                if self.obj_in_hand[obj_name]:
                    self.stage_checkpoints[obj_name + "_pickup"] = True
                    return 1
            if not self.stage_checkpoints[obj_name + "_thaw"]:
                if self.obj_freeze_state[obj_name] == 0:
                    self.stage_checkpoints[obj_name + "_thaw"] = True
                    return 1
        if not self.stage_checkpoints["succeed"]:
            if self._end_conditions():
                self.stage_checkpoints["succeed"] = True
                return 1
        self.stage_completion_tracker -= 1
        return 0

    def hand_crafted_policy(self):
        """
        A hand-crafted function to select action for next step
        Navigation is accurate
        """
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        fish = self.objs["fish"][0]
        fish_inhand = self.obj_in_hand["fish"]

        # Open the frig
        if not self.frig_open:
            if Open(self).can(self.electric_refrigerator):
                action = self.actions.open
            else:
                action = self.navigate_to(self.electric_refrigerator.cur_pos)
        # If any one of the object is in frig, we go to the frig and pick it up
        elif fish.check_rel_state(self, self.electric_refrigerator, 'inside'):
            if Pickup(self).can(fish):
                action = self.actions.pickup
            else:
                action = self.navigate_to(fish.cur_pos)
        elif self.sink in fwd_cell[0]:  # refrig should be in all three dimensions, sink is just in the first dimension
            if fish_inhand:
                if Drop(self).can(fish):
                    action = self.actions.drop_fish
                else:
                    self.adjusted_sink_pos = np.array(self.sink.cur_pos) + np.array([1, 0])
                    action = self.navigate_to(self.adjusted_sink_pos)
            else:
                # We're done, navigate randomly
                action = self.sample_nav_action()
        else:
            action = self.navigate_to(self.sink.cur_pos)

        return action

    def hand_crafted_lower_policy(self):
        """
        A hand-crafted function to select action for next step
        Navigation is accurate
        """
        num_factors = 3 + len(self.obj_name_list)
        factor = self.desired_goal[:num_factors].argmax()
        parent = self.desired_goal[num_factors:num_factors + 3 * (num_factors + 1)]
        parent = parent.reshape(num_factors + 1, 3).argmax(axis=-1)
        num_actions = len(self.actions)
        action = self.action_space.sample()

        if np.random.random() < 0.2:
            return action

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # possible_graphs
        agent_still = np.zeros_like(parent)
        agent_still[0] = 1
        agent_move = np.zeros_like(parent)
        agent_move[0] = agent_move[-1] = 1
        agent_blocked_frig = np.zeros_like(parent)
        agent_blocked_frig[0] = agent_move[1] = 1
        agent_blocked_sink = np.zeros_like(parent)
        agent_blocked_sink[0] = agent_move[2] = 1
        frig_by_agent = np.zeros_like(parent)
        frig_by_agent[0] = frig_by_agent[1]  = frig_by_agent[-1] = 1
        obj_by_agent = np.zeros_like(parent)
        obj_by_agent[0] = obj_by_agent[factor] = obj_by_agent[-1] = 1
        obj_freeze = np.zeros_like(parent)
        obj_freeze[1] = obj_freeze[factor] = 1
        obj_thaw = np.zeros_like(parent)
        obj_thaw[2] = obj_thaw[factor] = 1

        frig_pos = np.array(self.electric_refrigerator.cur_pos) + np.random.randint(0, 2, size=2)
        if factor == 0:
            if np.all(parent == agent_still):
                action = np.random.randint(3, num_actions)
            elif np.all(parent == agent_move):
                goal = np.random.randint(1, self.room_size - 1, size=2)
                action = self.navigate_to(goal)
                if self.electric_refrigerator in fwd_cell[0] and np.random.random() < 0.5:
                    action = np.random.choice([self.actions.forward, self.actions.open, self.actions.close])
                elif self.sink in fwd_cell[0] and np.random.random() < 0.5:
                    action = self.actions.forward
            elif np.all(parent == agent_blocked_frig):
                if self.electric_refrigerator in fwd_cell[0]:
                    action = self.actions.forward
                    if np.random.random() < 0.5:
                        action = np.random.choice([self.actions.open, self.actions.close])
                else:
                    action = self.navigate_to(frig_pos)
            elif np.all(parent == agent_blocked_sink):
                if self.sink in fwd_cell[0]:
                    action = self.actions.forward
                else:
                    action = self.navigate_to(self.sink.cur_pos)
        elif factor == 1:   # interact with the frig
            if np.all(parent == frig_by_agent):
                if self.electric_refrigerator in fwd_cell[0]:
                    if self.electric_refrigerator.check_abs_state(self, 'openable'):
                        action = self.actions.close
                        if np.random.random() < 0.2:
                            action = self.actions.pickup
                    else:
                        action = self.actions.open
                else:
                    action = self.navigate_to(frig_pos)
        elif factor > 2:    # interact with the object
            obj_id = factor - 3
            obj_name = self.obj_name_list[obj_id]
            obj = self.objs[obj_name][0]
            if np.all(parent == obj_by_agent):
                if obj in fwd_cell[0]:
                    action = self.actions.pickup
                else:
                    action = self.navigate_to(obj.cur_pos)
            elif np.all(parent == obj_thaw):
                if obj.check_abs_state(self, 'inhandofrobot'):
                    if self.sink in fwd_cell[0]:
                        action = self.actions["drop_" + obj_name]
                    else:
                        action = self.navigate_to(self.sink.cur_pos)
                else:
                    if obj in fwd_cell[0]:
                        action = self.actions.pickup
                    else:
                        action = self.navigate_to(obj.cur_pos)
            elif np.all(parent == obj_freeze):
                if obj.check_abs_state(self, 'inhandofrobot'):
                    if self.electric_refrigerator in fwd_cell[0]:
                        action = self.actions["drop_" + obj_name]
                    else:
                        action = self.navigate_to(self.electric_refrigerator.cur_pos)
                else:
                    if obj in fwd_cell[0]:
                        action = self.actions.pickup
                    else:
                        action = self.navigate_to(obj.cur_pos)

        return action

    def gen_obs(self):
        self.electric_refrigerator = self.objs['electric_refrigerator'][0]
        self.sink = self.objs['sink'][0]
        self.frig_open = int(self.electric_refrigerator.check_abs_state(self, 'openable'))

        obs = {
            "agent": np.array([*self.agent_pos, self.agent_dir]),
            "sink": np.array(self.sink.cur_pos),
            "frig": np.array([*self.electric_refrigerator.cur_pos, self.frig_open]),
        }

        self.obj_in_hand = {}
        self.obj_freeze_state = {}
        for obj_name in self.obj_name_list:
            obj = self.objs[obj_name][0]
            self.obj_freeze_state[obj_name] = obj_frozen_state = int(obj.check_abs_state(self, 'freezable'))
            self.obj_in_hand[obj_name] = obj_inhand = int(obj.check_abs_state(self, 'inhandofrobot'))
            obj_pos = np.array(self.agent_pos) if obj_inhand else np.array(obj.cur_pos)
            obs[obj_name] = np.array([*obj_pos, obj_frozen_state])

        if not self.discrete_obs:
            for k, v in obs.items():
                obs[k] = (2. * v / (self.original_observation_space[k].nvec - 1) - 1).astype(np.float32)

        return obs

    def step(self, action, evaluate_mask=True):
        prev_freeze_state = deepcopy(self.obj_freeze_state)

        self.update_states()
        if self.desired_goal is not None:
            action = self.hand_crafted_lower_policy()

        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        frig_manipulated = pickup_done = drop_done = False
        pickup_obj_name = drop_obj_name = None

        # Rotate left
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            can_overlap = True
            obstacle = None
            for dim in fwd_cell:
                for obj in dim:
                    if is_obj(obj) and not obj.can_overlap:
                        can_overlap = False
                        obstacle = obj
                        break

            if can_overlap:
                self.agent_pos = fwd_pos
        elif action == self.actions.open:
            if Open(self).can(self.electric_refrigerator):
                Open(self).do(self.electric_refrigerator)
                frig_manipulated = True
        elif action == self.actions.close:
            if Close(self).can(self.electric_refrigerator):
                Close(self).do(self.electric_refrigerator)
                frig_manipulated = True
        elif action == self.actions.pickup:
            for obj_name in self.obj_name_list:
                obj = self.objs[obj_name][0]
                if Pickup(self).can(obj):
                    Pickup(self).do(obj)
                    pickup_done, pickup_obj_name = True, obj_name
                    break
        else:
            action_name = self.actions(action).name
            assert action_name.startswith("drop")
            drop_obj_name = self.actions(action).name.split('_')[1]
            drop_obj = self.objs[drop_obj_name][0]
            if Drop(self).can(drop_obj):
                drop_done = True
                self.drop_rand_dim(drop_obj)

        obs = self.gen_obs()
        reward = self._reward()
        terminated = False  # self._end_conditions()
        truncated = self.step_count >= self.max_steps
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        if evaluate_mask:
            feature_dim = 8 + 3 * len(self.obj_name_list)
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            frig_pos_idxes = slice(3, 5)
            frig_state_idx = 5
            sink_pos_idxes = slice(6, 8)

            start_idx = 8
            obj_pos_slices = {}
            obj_state_idxes = {}
            for obj_name in self.obj_name_list:
                obj_pos_slices[obj_name] = slice(start_idx, start_idx + 2)
                obj_state_idxes[obj_name] = start_idx + 2
                start_idx += 3

            action_idx = start_idx

            def extract_obj_pos_idxes(obj_):
                if obj_ == self.electric_refrigerator:
                    return frig_pos_idxes
                elif obj_ == self.sink:
                    return sink_pos_idxes
                else:
                    for obj_name in self.obj_name_list:
                        if obj_ == self.objs[obj_name][0]:
                            return obj_pos_slices[obj_name]
                    return None

            if action == self.actions.left or action == self.actions.right:
                mask[agent_dir_idx, action_idx] = True

            # Move forward
            elif action == self.actions.forward:
                pos_idx = self.agent_dir % 2
                if can_overlap:
                    mask[pos_idx, agent_dir_idx] = True
                    mask[pos_idx, action_idx] = True
                    for obj_name, obj_in_hand in self.obj_in_hand.items():
                        if obj_in_hand:
                            obj_pos_idxes = obj_pos_slices[obj_name]
                            obj_change_pos_idxes = obj_pos_idxes.start + pos_idx
                            mask[obj_change_pos_idxes, agent_pos_idxes] = True
                            mask[obj_change_pos_idxes, agent_dir_idx] = True
                            mask[obj_change_pos_idxes, obj_change_pos_idxes] = True
                            mask[obj_change_pos_idxes, action_idx] = True
                else:
                    mask[pos_idx, agent_pos_idxes] = True
                    mask[pos_idx, agent_dir_idx] = True
                    obstacle_pos_idxes = extract_obj_pos_idxes(obj)
                    if obstacle_pos_idxes is not None:
                        mask[pos_idx, obstacle_pos_idxes] = True
                    for obj_name, obj_in_hand in self.obj_in_hand.items():
                        if obj_in_hand:
                            obj_pos_idxes = obj_pos_slices[obj_name]
                            obj_change_pos_idxes = obj_pos_idxes.start + pos_idx
                            mask[obj_change_pos_idxes, agent_pos_idxes] = True
                            mask[obj_change_pos_idxes, agent_dir_idx] = True
                            mask[obj_change_pos_idxes, obj_change_pos_idxes] = True
                            if obstacle_pos_idxes is not None:
                                mask[obj_change_pos_idxes, obstacle_pos_idxes] = True

            if pickup_done:
                obj_pos_idxes = obj_pos_slices[pickup_obj_name]
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True

            if drop_done:
                obj_pos_idxes = obj_pos_slices[drop_obj_name]
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True
                inside = drop_obj.inside_of
                if inside is not None and inside == self.electric_refrigerator:
                    mask[obj_pos_idxes, frig_pos_idxes] = True
                    mask[obj_pos_idxes, frig_state_idx] = True

            if frig_manipulated:
                mask[frig_state_idx, action_idx] = True
                mask[frig_state_idx, agent_pos_idxes] = True
                mask[frig_state_idx, agent_dir_idx] = True
                mask[frig_state_idx, frig_pos_idxes] = True

            # update freeze mask
            for obj_name in self.obj_name_list:
                prev_obj_freeze_state = prev_freeze_state[obj_name]
                cur_obj_freeze_state = self.obj_freeze_state[obj_name]
                obj_pos_idxes = obj_pos_slices[obj_name]
                obj_state_idx = obj_state_idxes[obj_name]
                if cur_obj_freeze_state > prev_obj_freeze_state:
                    mask[obj_state_idx, obj_pos_idxes] = True
                    mask[obj_state_idx, frig_pos_idxes] = True
                elif cur_obj_freeze_state < prev_obj_freeze_state:
                    mask[obj_state_idx, obj_pos_idxes] = True
                    mask[obj_state_idx, sink_pos_idxes] = True
            info["variable_graph"] = mask

            num_factors = 3 + len(self.obj_name_list)
            agent_idxes = slice(0, 3)
            frig_idxes = slice(3, 6)
            sink_idxes = slice(6, 8)
            start_idx = 8
            obj_idxes = []
            for _ in self.obj_name_list:
                obj_idxes.append(slice(start_idx, start_idx + 3))
                start_idx += 3
            action_idx = start_idx

            factor_mask = np.zeros((num_factors, num_factors + 1), dtype=bool)
            for i, idxes in enumerate([agent_idxes, frig_idxes, sink_idxes] + obj_idxes):
                for j, pa_idxes in enumerate([agent_idxes, frig_idxes, sink_idxes] + obj_idxes + [action_idx]):
                    factor_mask[i, j] = mask[idxes, pa_idxes].any()
            info["factor_graph"] = factor_mask

        return obs, reward, terminated, truncated, info


register(
    id='MiniGrid-thawing-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv',
    kwargs={}
)
