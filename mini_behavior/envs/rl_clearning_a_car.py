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
from .cleaning_a_car import CleaningACarEnv


# TODO: fix action and step()
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
        open = 3
        close = 4
        pickup_rag = 5
        drop_rag = 6
        toggle_sink = 7
        pickup_soap = 8
        drop_soap = 9
        drop_date = 10

    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=100,
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
                action = self.actions.drop_fish
            elif self.date_inhand:
                action = self.actions.drop_date
            elif self.olive_inhand:
                action = self.actions.drop_olive
            else:
                # We're done, navigate randomly
                action = self.sample_nav_action()
        else:
            action = self.navigate_to(np.array(self.sink.cur_pos))

        return action

    def gen_obs(self):

        self.car = self.objs['car'][0]
        self.rag = self.objs['rag'][0]
        self.shelf = self.objs['shelf'][0]
        self.soap = self.objs['soap'][0]
        self.bucket = self.objs['bucket'][0]
        self.sink = self.objs['sink'][0]
        # The state that we need:
        # Car: Do you need openable? Why would you want to open the car?
        self.car_stain = int(self.car.check_abs_state(self, 'stainable'))
        # rag: soak, cleaness, {inside (bucket), onTop (of car), inhand}?
        self.rag_soak = int(self.rag.check_abs_state(self, 'soakable'))
        self.rag_cleanness = int(self.rag.check_abs_state(self, 'cleanness'))
        self.sink_toggled = int(self.sink.check_abs_state(self, 'toggleable'))



        obs = {
            "agent_pos": np.array(self.agent_pos),
            "agent_dir": self.agent_dir,
            "car_pos": np.array(self.car.cur_pos),
            "car_state": np.array([self.car_stain]),
            "bucket_pos": np.array(self.bucket.cur_pos),
            "soap_pos": np.array([self.soap.cur_pos]),
            "sink_pos": np.array(self.date.cur_pos),
            "sink_state": np.array([self.sink_toggled]),
            "bucket_pos": np.array(self.bucket.cur_pos),
            "rag_pos": np.array(self.rag.cur_pos),
            "rag_state": np.array([self.rag_soak, self.rag_cleanness])
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


register(
    id='MiniGrid-SimpleCleaningACarEnv-16x16-N2-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv'
)

register(
    id='MiniGrid-SimpleCleaningACarEnv-8x8-N2-v0',
    entry_point='mini_behavior.envs:SimpleThawingFrozenFoodEnv',
    kwargs={'room_size': 8}
)