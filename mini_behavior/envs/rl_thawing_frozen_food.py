from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.grid import is_obj
from mini_behavior.actions import Pickup, Drop, Toggle, Open, Close
from mini_behavior.objects import Wall
from bddl import ACTION_FUNC_MAPPING
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
            max_steps=150,
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

    def get_observation_dims(self):
        return {
            "agent_pos": np.array([self.room_size, self.room_size]),
            "agent_dir": np.array([4]),
            "fish_pos": np.array([self.room_size, self.room_size]),
            "fish_state": np.array([6, 2, 2, 2]),
            "olive_pos": np.array([self.room_size, self.room_size]),
            "olive_state": np.array([6, 2, 2, 2]),
            "date_pos": np.array([self.room_size, self.room_size]),
            "date_state": np.array([6, 2, 2, 2]),
            "sink_pos": np.array([self.room_size, self.room_size]),
            "frig_pos": np.array([self.room_size, self.room_size]),
            "frig_state": np.array([2])
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
        Notice that navigation is still random
        """
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # If any one of the object is not in frig, we go to the frig and drop it
        # test navigate to frig
        return self.navigate_to(np.array(self.electric_refrigerator.cur_pos), np.array(self.agent_pos), self.agent_dir)

        # if not self.frig_open and Open(self).can(self.electric_refrigerator):
        #     action = self.actions.open
        # # if any of the objects is still in frig (Ok the agent is going to get blocked by frig...)
        # elif
        # elif self.printer_inhandofrobot:
        #     if self.table in fwd_cell[1]:
        #         action = 4  # drop
        #     else:
        #         action = self.sample_nav_action()
        #         action = self.navigate_to(np.array(self.electric_refrigerator.cur_pos), np.array(self.agent_pos), self.agent_dir)
        #
        # elif not self.printer_ontop_table and Pickup(self).can(self.printer):
        #     action = 3
        #
        # else:
        #     action = self.sample_nav_action()
        #
        # return action

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


        obs = {
            "agent_pos": np.array(self.agent_pos),
            "agent_dir": self.agent_dir,
            "fish_pos": np.array(self.fish.cur_pos),
            "fish_state": np.array([self.fish_frozen, self.fish_inhand, self.fish_inside, self.fish_nextto]),
            "olive_pos": np.array(self.olive.cur_pos),
            "olive_state": np.array([self.olive_frozen, self.olive_inhand, self.olive_inside, self.olive_nextto]),
            "date_pos": np.array(self.date.cur_pos),
            "date_state": np.array([self.date_frozen, self.date_inhand, self.date_inside, self.date_nextto]),
            "sink_pos": np.array(self.sink.cur_pos),
            "frig_pos": np.array(self.electric_refrigerator.cur_pos),
            "frig_state": np.array([self.frig_open])
        }

        return obs

    def step(self, action):
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
        elif action == self.actions.pickup_date:
            if Pickup(self).can(self.date):
                Pickup(self).do(self.date)
        elif action == self.actions.pickup_olive:
            if Pickup(self).can(self.olive):
                Pickup(self).do(self.olive)
        elif action == self.actions.pickup_fish:
            if Pickup(self).can(self.fish):
                Pickup(self).do(self.fish)
        # Drop dimension: 2 is the top
        elif action == self.actions.drop_date:
            if Drop(self).can(self.date):
                Drop(self).do(self.date, 2)
        elif action == self.actions.drop_olive:
            if Drop(self).can(self.olive):
                Drop(self).do(self.olive, 2)
        elif action == self.actions.drop_fish:
            if Drop(self).can(self.fish):
                Drop(self).do(self.fish, 2)
        elif action == self.actions.open:
            if Open(self).can(self.electric_refrigerator):
                Open(self).do(self.electric_refrigerator)
        elif action == self.actions.close:
            if Close(self).can(self.electric_refrigerator):
                Close(self).do(self.electric_refrigerator)

        self.update_states()
        reward = self._reward()
        done = self._end_conditions() or self.step_count >= self.max_steps
        obs = self.gen_obs()

        return obs, reward, done, {}

    def _reward(self):
        if self._end_conditions():
            self.reward += 100
        else:
            self.reward -= 1

        return self.reward


register(
    id='MiniGrid-SimpleThawingFrozenFoodEnv-16x16-N2-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv'
)

register(
    id='MiniGrid-SimpleThawingFrozenFoodEnv-8x8-N2-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv',
    kwargs={'room_size': 8}
)