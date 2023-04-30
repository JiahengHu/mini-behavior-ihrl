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



class SimpleThawingFrozenFoodEnv(ThawingFrozenFoodEnv):
    """
    Environment in which the agent is instructed to ...
    This is a wrapper around the original mini-behavior environment where states are represented by category, and
    actions are converted to integer selection
    """
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        open = 3
        close = 4
        pickup_fish = 5
        pickup_olive = 6
        pickup_date = 7
        drop_fish = 8
        drop_olive = 9
        drop_date = 10

    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=300,
    ):
        self.room_size = room_size

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        # We redefine action space here
        self.actions = SimpleThawingFrozenFoodEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_dim = len(self.actions)

        self.reward_range = (-math.inf, math.inf)
        self.init_stage_checkpoint()

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"frig_open": False, "date_thaw": False, "fish_thaw": False,
                                  "olive_thaw": False, "succeed": False}
        self.stage_completion_tracker = 0

    def update_stage_checkpoint(self):
        self.stage_completion_tracker += 1
        if not self.stage_checkpoints["frig_open"]:
            if self.frig_open:
                self.stage_checkpoints["frig_open"] = True
                return 1
        if not self.stage_checkpoints["date_thaw"]:
            if self.date_frozen == 0:
                self.stage_checkpoints["date_thaw"] = True
                return 1
        if not self.stage_checkpoints["fish_thaw"]:
            if self.fish_frozen == 0:
                self.stage_checkpoints["fish_thaw"] = True
                return 1
        if not self.stage_checkpoints["olive_thaw"]:
            if self.olive_frozen == 0:
                self.stage_checkpoints["olive_thaw"] = True
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
        return {
            "agent_pos": np.array([self.room_size, self.room_size]),
            "agent_dir": np.array([4]),
            "fish_pos": np.array([self.room_size, self.room_size]),
            "fish_state": np.array([6]),
            "olive_pos": np.array([self.room_size, self.room_size]),
            "olive_state": np.array([6]),
            "date_pos": np.array([self.room_size, self.room_size]),
            "date_state": np.array([6]),
            "sink_pos": np.array([self.room_size, self.room_size]),
            "frig_pos": np.array([self.room_size, self.room_size]),
            "frig_state": np.array([2]),
            "step_count": np.array([1])
        }

    def generate_action(self):
        # probability of choosing the hand-crafted action
        prob = 1.0  # 1.0
        if self.np_random.random() < prob:
            return self.hand_crafted_policy()
        else:
            return self.action_space.sample()

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
        elif self.olive_inside:
            if Pickup(self).can(self.olive):
                action = self.actions.pickup_olive
            else:
                action = self.navigate_to(np.array(self.olive.cur_pos))
        elif self.fish_inside:
            if Pickup(self).can(self.fish):
                action = self.actions.pickup_fish
            else:
                action = self.navigate_to(np.array(self.fish.cur_pos))
        elif self.date_inside:
            if Pickup(self).can(self.date):
                action = self.actions.pickup_date
            else:
                action = self.navigate_to(np.array(self.date.cur_pos))
        elif self.sink in fwd_cell[0]:  # refrig should be in all three dimensions, sink is just in the first dimension
            if self.fish_inhand:
                if Drop(self).can(self.fish):
                    action = self.actions.drop_fish
                else:
                    self.adjusted_sink_pos = np.array(self.sink.cur_pos) + np.array([1, 0])
                    action = self.navigate_to(self.adjusted_sink_pos)
            elif self.date_inhand:
                if Drop(self).can(self.date):
                    action = self.actions.drop_date
                else:
                    self.adjusted_sink_pos = np.array(self.sink.cur_pos) + np.array([1, 0])
                    action = self.navigate_to(self.adjusted_sink_pos)
            elif self.olive_inhand:
                if Drop(self).can(self.olive):
                    action = self.actions.drop_olive
                else:
                    self.adjusted_sink_pos = np.array(self.sink.cur_pos) + np.array([1, 0])
                    action = self.navigate_to(self.adjusted_sink_pos)
            else:
                # We're done, navigate randomly
                action = self.sample_nav_action()
        else:
            action = self.navigate_to(np.array(self.adjusted_sink_pos))

        return action

    def gen_obs(self):
        self.date = self.objs['date'][0]
        self.olive = self.objs['olive'][0]
        self.fish = self.objs['fish'][0]
        self.electric_refrigerator = self.objs['electric_refrigerator'][0]
        self.sink = self.objs['sink'][0]

        # The state that we need: open / close, frozen / not & in hand & nextTo & inside (x3) [Thats it]
        self.olive_frozen = int(self.olive.check_abs_state(self, 'freezable'))
        self.olive_inhand = int(self.olive.check_abs_state(self, 'inhandofrobot'))
        self.olive_inside = int(self.olive.check_rel_state(self, self.electric_refrigerator, 'inside'))
        self.olive_nextto = int(self.olive.check_rel_state(self, self.sink, 'nextto'))

        self.fish_frozen = int(self.fish.check_abs_state(self, 'freezable'))
        self.fish_inhand = int(self.fish.check_abs_state(self, 'inhandofrobot'))
        self.fish_inside = int(self.fish.check_rel_state(self, self.electric_refrigerator, 'inside'))
        self.fish_nextto = int(self.fish.check_rel_state(self, self.sink, 'nextto'))

        self.date_frozen = int(self.date.check_abs_state(self, 'freezable'))
        self.date_inhand = int(self.date.check_abs_state(self, 'inhandofrobot'))
        self.date_inside = int(self.date.check_rel_state(self, self.electric_refrigerator, 'inside'))
        self.date_nextto = int(self.date.check_rel_state(self, self.sink, 'nextto'))

        self.frig_open = int(self.electric_refrigerator.check_abs_state(self, 'openable'))

        # We removed "inhand, inside, and nexto" since they can be inferred from position
        # e.g.: , self.fish_inhand, self.fish_inside, self.fish_nextto

        obs = {
            "agent_pos": np.array(self.agent_pos),
            "agent_dir": np.array([self.agent_dir]),
            "fish_pos": np.array(self.fish.cur_pos),
            "fish_state": np.array([self.fish_frozen]),
            "olive_pos": np.array(self.olive.cur_pos),
            "olive_state": np.array([self.olive_frozen]),
            "date_pos": np.array(self.date.cur_pos),
            "date_state": np.array([self.date_frozen]),
            "sink_pos": np.array(self.sink.cur_pos),
            "frig_pos": np.array(self.electric_refrigerator.cur_pos),
            "frig_state": np.array([self.frig_open]),
            "step_count": np.array([float(self.step_count) / self.max_steps])
        }

        return obs

    def step(self, action, evaluate_mask=True):
        cur_olive_frozen, cur_fish_frozen, cur_date_frozen = self.olive_frozen, self.fish_frozen, self.date_frozen

        self.update_states()
        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        pickup_done = frig_manipulated = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

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

        elif action == self.actions.pickup_date:
            if Pickup(self).can(self.date):
                Pickup(self).do(self.date)
                pickup_done, obj = True, self.date
        elif action == self.actions.pickup_olive:
            if Pickup(self).can(self.olive):
                Pickup(self).do(self.olive)
                pickup_done, obj = True, self.olive
        elif action == self.actions.pickup_fish:
            if Pickup(self).can(self.fish):
                Pickup(self).do(self.fish)
                pickup_done, obj = True, self.fish
        # Drop dimension: 2 is the top
        elif action == self.actions.drop_date:
            obj = self.date
            self.drop_rand_dim(obj)
        elif action == self.actions.drop_olive:
            obj = self.olive
            self.drop_rand_dim(obj)
        elif action == self.actions.drop_fish:
            obj = self.fish
            self.drop_rand_dim(obj)
        elif action == self.actions.open:
            if Open(self).can(self.electric_refrigerator):
                Open(self).do(self.electric_refrigerator)
                frig_manipulated = True
        elif action == self.actions.close:
            if Close(self).can(self.electric_refrigerator):
                Close(self).do(self.electric_refrigerator)
                frig_manipulated = True

        reward = self._reward()
        # done = self._end_conditions() or self.step_count >= self.max_steps
        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        if evaluate_mask:
            feature_dim = 17
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            fish_pos_idxes = slice(3, 5)
            fish_state_idx = 5
            olive_pos_idxes = slice(6, 8)
            olive_state_idx = 8
            date_pos_idxes = slice(9, 11)
            date_state_idx = 11
            sink_pos_idxes = slice(12, 14)
            frig_pos_idxes = slice(14, 16)
            frig_state_idx = 16
            action_idx = 17

            def extract_obj_pos_idxes(obj_):
                if obj == self.fish:
                    return fish_pos_idxes
                elif obj == self.olive:
                    return olive_pos_idxes
                elif obj == self.date:
                    return date_pos_idxes
                elif obj == self.sink:
                    return sink_pos_idxes
                elif obj == self.electric_refrigerator:
                    return frig_pos_idxes
                else:
                    return None

            if action == self.actions.left or action == self.actions.right:
                mask[agent_dir_idx, action_idx] = True

            # Move forward
            elif action == self.actions.forward:
                pos_idx = self.agent_dir % 2
                if can_overlap:
                    mask[pos_idx, agent_dir_idx] = True
                    mask[pos_idx, action_idx] = True
                else:
                    mask[pos_idx, agent_pos_idxes] = True
                    mask[pos_idx, agent_dir_idx] = True
                    obj_pos_idxes = extract_obj_pos_idxes(obj)
                    if obj_pos_idxes is not None:
                        mask[pos_idx, obj_pos_idxes] = True

            if pickup_done:
                obj_pos_idxes = extract_obj_pos_idxes(obj)
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                mask[obj_pos_idxes, obj_pos_idxes] = True
                mask[obj_pos_idxes, action_idx] = True

            if action in [self.actions.drop_date, self.actions.drop_olive, self.actions.drop_fish]:
                obj_pos_idxes = extract_obj_pos_idxes(obj)
                mask[obj_pos_idxes, agent_pos_idxes] = True
                mask[obj_pos_idxes, agent_dir_idx] = True
                if not obj.check_abs_state(self, 'inhandofrobot'):
                    mask[obj_pos_idxes, action_idx] = True

            if frig_manipulated:
                mask[frig_state_idx, action_idx] = True
                mask[frig_state_idx, agent_pos_idxes] = True
                mask[frig_state_idx, agent_dir_idx] = True
                mask[frig_state_idx, frig_pos_idxes] = True

            # update freeze mask
            for cur_obj_frozen, next_obj_frozen, obj_pos_idxes, obj_state_idx in \
                [[cur_olive_frozen, self.olive_frozen, olive_pos_idxes, olive_state_idx],
                 [cur_fish_frozen, self.fish_frozen, fish_pos_idxes, fish_state_idx],
                 [cur_date_frozen, self.date_frozen, date_pos_idxes, date_state_idx]
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
