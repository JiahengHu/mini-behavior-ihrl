from copy import deepcopy
from enum import IntEnum

import math
from gymnasium import spaces

from mini_behavior.actions import Pickup, Drop, DropIn, Toggle
from mini_behavior.floorplan import *
from mini_behavior.grid import is_obj
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
        pickup = 3
        toggle = 4      # sink
        drop_soap = 5
        drop_rag = 6


    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=300,
            use_stage_reward=False,
            seed=42,
            evaluate_graph=False,
            random_obj_pose=True,
            discrete_obs=True,
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward
        self.evaluate_graph = evaluate_graph
        self.random_obj_pose = random_obj_pose
        self.discrete_obs = discrete_obs

        # state space
        self.original_observation_space = spaces.Dict([
                ("agent", spaces.MultiDiscrete([self.room_size, self.room_size, 4])),
                ("car", spaces.MultiDiscrete([self.room_size, self.room_size, 2])),
                ("bucket", spaces.MultiDiscrete([self.room_size, self.room_size, 2])),
                ("soap", spaces.MultiDiscrete([self.room_size, self.room_size])),
                ("sink", spaces.MultiDiscrete([self.room_size, self.room_size, 2])),
                ("rag", spaces.MultiDiscrete([self.room_size, self.room_size, 6, 6])),
            ])

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         seed=seed
                         )

        # action space
        self.actions = SimpleCleaningACarEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_dim = len(self.actions)

        # override observation space
        if self.discrete_obs:
            self.observation_space = self.original_observation_space
        else:
            self.observation_space = spaces.Dict([(name, spaces.Box(low=-1, high=1, shape=[len(space.nvec)]))
                                                  for name, space in self.original_observation_space.spaces.items()])

        self.reward_range = (-math.inf, math.inf)
        self.init_stage_checkpoint()

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"rag_soaked": False, "car_not_stain": False, "succeed": False}
        self.stage_completion_tracker = 0

    def _gen_objs(self):
        if self.random_obj_pose:
            return super()._gen_objs()
        else:
            car = self.objs['car'][0]
            shelf = self.objs['shelf'][0]
            bucket = self.objs['bucket'][0]
            sink = self.objs['sink'][0]
            rag = self.objs['rag'][0]
            soap = self.objs['soap'][0]

            assert self.room_size >= 10

            car_top = (1, 1)
            self.put_obj(car, *car_top, 0)

            sink_top = (2, 6)
            self.put_obj(sink, *sink_top, 0)

            shelf_top = (6, 5)
            self.put_obj(shelf, *shelf_top, 0)

            bucket_top = (7, 2)
            self.put_obj(bucket, *bucket_top, 0)

            shelf_pos_idx = 0
            for obj in [rag, soap]:
                self.put_obj(obj, *shelf.all_pos[shelf_pos_idx], 2)
                obj.states['inside'].set_value(shelf, True)
                shelf_pos_idx += 1

            # rag not soaked
            rag.states['soakable'].set_value(0)

            # dusty car
            car.states['stainable'].set_value(True)

    def place_agent(self):
        if self.random_obj_pose:
            return super().place_agent()
        else:
            self.agent_pos = np.array([4, 4])
            self.agent_dir = 0
            return self.agent_pos

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.init_stage_checkpoint()
        return obs, info

    def _init_conditions(self):
        super()._init_conditions()
        bucket = self.objs['bucket'][0]
        assert bucket.check_abs_state(self, "soapable") == False

        return True

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

        bucket_soaped = self.bucket.check_abs_state(self, "soapable")

        # for compute ground truth causal graph
        self.obj_state = {
            "car_stain": self.car_stain,
            "rag_soak": self.rag_soak,
            "rag_cleanness": self.rag_cleanness,
            "bucket_soaped": bucket_soaped,
        }

        obs = {
            "agent": np.array([*self.agent_pos, self.agent_dir]),
            "car": np.array([*self.car.cur_pos, self.car_stain]),
            "bucket": np.array([*self.bucket.cur_pos, bucket_soaped]),
            "soap": np.array([*self.soap.cur_pos]),
            "sink": np.array([*self.sink.cur_pos, self.sink_toggled]),
            "rag": np.array([*self.rag.cur_pos, self.rag_soak, self.rag_cleanness]),
        }

        self.obj_in_hand = {}
        for obj_name in ["rag", "soap"]:
            obj = self.objs[obj_name][0]
            self.obj_in_hand[obj_name] = obj_inhand = int(obj.check_abs_state(self, 'inhandofrobot'))
            obj_pos = np.array(self.agent_pos) if obj_inhand else np.array(obj.cur_pos)
            obs[obj_name][:2] = obj_pos

        if not self.discrete_obs:
            for k, v in obs.items():
                obs[k] = (2. * v / (self.original_observation_space[k].nvec - 1) - 1).astype(np.float32)

        return obs

    def step(self, action):
        prev_obj_state = deepcopy(self.obj_state)
        self.update_states()

        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        sink_manipulated = pickup_succeed = drop_succeed = False
        pickup_obj_name = drop_obj_name = obstacle = None

        pickable_obj_names = ['rag', 'soap']

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
                        obstacle = obj
                        break
            if can_overlap:
                self.agent_pos = fwd_pos
        elif action == self.actions.pickup:
            for obj_name in pickable_obj_names:
                obj = self.objs[obj_name][0]
                if Pickup(self).can(obj):
                    Pickup(self).do(obj)
                    pickup_succeed, pickup_obj_name = True, obj_name
                    break
        elif action == self.actions.toggle:
            if Toggle(self).can(self.sink):
                Toggle(self).do(self.sink)
                sink_manipulated = True
        else:
            action_name = self.actions(action).name
            assert action_name.startswith("drop")
            drop_obj_name = self.actions(action).name.split('_')[1]
            drop_obj = self.objs[drop_obj_name][0]
            if DropIn(self).can(drop_obj):
                drop_succeed = True
                DropIn(self).do(drop_obj, np.random.choice(drop_obj.available_dims))
            elif Drop(self).can(drop_obj):
                drop_succeed = True
                Drop(self).do(drop_obj, np.random.choice(drop_obj.available_dims))

        obs = self.gen_obs()
        reward = self._reward()
        terminated = False  # self._end_conditions()
        truncated = self.step_count >= self.max_steps
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        if self.evaluate_graph:
            num_variables = 18
            mask = np.eye(num_variables, num_variables + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            car_pos_idxes = slice(3, 5)
            car_state_idx = 5
            bucket_pos_idxes = slice(6, 8)
            bucket_state_idx = 8
            soap_pos_idxes = slice(9, 11)
            sink_pos_idxes = slice(11, 13)
            sink_state_idx = 13
            rag_pos_idxes = slice(14, 16)
            rag_soak_state_idx = 16
            rag_clean_state_idx = 17

            action_idx = num_variables

            obj_pos_slices = {
                "car": car_pos_idxes,
                "bucket": bucket_pos_idxes,
                "soap": soap_pos_idxes,
                "sink": sink_pos_idxes,
                "rag": rag_pos_idxes,
            }

            def extract_obj_pos_idxes(obj_):
                for obj_name, pos_idxes in obj_pos_slices.items():
                    if obj_ == self.objs[obj_name][0]:
                        return pos_idxes
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
                    obstacle_pos_idxes = extract_obj_pos_idxes(obstacle)
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

            if pickup_succeed:
                obj_pos_idxes = obj_pos_slices[pickup_obj_name]
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True

            if drop_succeed:
                obj_pos_idxes = obj_pos_slices[drop_obj_name]
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True

            if sink_manipulated:
                mask[sink_state_idx, action_idx] = True
                mask[sink_state_idx, agent_pos_idxes] = True
                mask[sink_state_idx, agent_dir_idx] = True
                mask[sink_state_idx, sink_pos_idxes] = True

            # update car cleaning mask
            if prev_obj_state["car_stain"] != self.obj_state["car_stain"]:
                mask[car_state_idx, car_pos_idxes] = True
                mask[car_state_idx, rag_pos_idxes] = True
                mask[car_state_idx, rag_soak_state_idx] = True
                mask[rag_clean_state_idx, car_pos_idxes] = True
                mask[rag_clean_state_idx, rag_pos_idxes] = True
                mask[rag_clean_state_idx, rag_soak_state_idx] = True
                if action == self.actions.drop_rag:
                    mask[car_state_idx, agent_pos_idxes] = True
                    mask[car_state_idx, agent_dir_idx] = True
                    mask[car_state_idx, action_idx] = True
                    mask[rag_clean_state_idx, agent_pos_idxes] = True
                    mask[rag_clean_state_idx, agent_dir_idx] = True
                    mask[rag_clean_state_idx, action_idx] = True

            # update rag soaking mask
            if prev_obj_state["rag_soak"] != self.obj_state["rag_soak"]:
                mask[rag_soak_state_idx, sink_pos_idxes] = True
                mask[rag_soak_state_idx, rag_pos_idxes] = True
                mask[rag_soak_state_idx, rag_soak_state_idx] = True
                if action == self.actions.drop_rag:
                    mask[rag_soak_state_idx, agent_pos_idxes] = True
                    mask[rag_soak_state_idx, agent_dir_idx] = True
                    mask[rag_soak_state_idx, action_idx] = True

            # update rag cleanness mask
            if prev_obj_state["rag_cleanness"] < self.obj_state["rag_cleanness"]:
                mask[rag_clean_state_idx, bucket_pos_idxes] = True
                mask[rag_clean_state_idx, bucket_state_idx] = True
                mask[rag_clean_state_idx, rag_pos_idxes] = True
                if action == self.actions.drop_rag:
                    mask[rag_clean_state_idx, agent_pos_idxes] = True
                    mask[rag_clean_state_idx, agent_dir_idx] = True
                    mask[rag_clean_state_idx, action_idx] = True

            # update bucket soaped mask
            if prev_obj_state["bucket_soaped"] < self.obj_state["bucket_soaped"]:
                mask[bucket_state_idx, bucket_pos_idxes] = True
                mask[bucket_state_idx, soap_pos_idxes] = True
                if action == self.actions.drop_soap:
                    mask[bucket_state_idx, agent_pos_idxes] = True
                    mask[bucket_state_idx, agent_dir_idx] = True
                    mask[bucket_state_idx, action_idx] = True

            num_factors = 6
            agent_idxes = slice(0, 3)
            car_idxes = slice(3, 6)
            bucket_idxes = slice(6, 9)
            soap_idxes = slice(9, 11)
            sink_idxes = slice(11, 14)
            rag_idxes = slice(14, 18)

            factor_idxes = [agent_idxes, car_idxes, bucket_idxes, soap_idxes, sink_idxes, rag_idxes]

            factor_mask = np.zeros((num_factors, num_factors + 1), dtype=bool)
            for i, idxes in enumerate(factor_idxes):
                for j, pa_idxes in enumerate(factor_idxes + [action_idx]):
                    factor_mask[i, j] = mask[idxes, pa_idxes].any()
            info["factor_graph"] = factor_mask

        return obs, reward, terminated, truncated, info

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
                    action = self.go_toggle(self.sink, self.actions.toggle)
                elif self.rag_soak != 5:
                    action = self.sample_nav_action()
                else:
                    action = self.go_pickup(self.rag, self.actions.pickup)
            else:
                action = self.go_pickup(self.rag, self.actions.pickup)
        else:
            rag_in_bucket = self.rag.check_rel_state(self, self.bucket, "atsamelocation")
            soap_in_bucket = self.soap.check_rel_state(self, self.bucket, "atsamelocation")
            if rag_in_bucket and soap_in_bucket:
                action = self.sample_nav_action()
            elif not soap_in_bucket:
                if self.soap.check_abs_state(self, 'inhandofrobot'):
                    action = self.go_drop(self.bucket, fwd_cell, 0, self.actions.drop_soap)
                else:
                    action = self.go_pickup(self.soap, self.actions.pickup)
            elif not rag_in_bucket:
                if self.rag.check_abs_state(self, 'inhandofrobot'):
                    action = self.go_drop(self.bucket, fwd_cell, 0, self.actions.drop_rag)
                else:
                    action = self.go_pickup(self.rag, self.actions.pickup)
            else:
                print("reaching here is impossible")
                raise NotImplementedError

        return action


register(
    id='MiniGrid-cleaning_car-v0',
    entry_point='mini_behavior.envs:SimpleCleaningACarEnv',
    kwargs={}
)
