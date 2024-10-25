# MODIFIED FROM MINIGRID REPO

import os
import pickle as pkl
import gymnasium as gym
import numpy as np

from enum import IntEnum
from gymnasium import spaces
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.mission import MissionSpace
from mini_bddl.actions import ACTION_FUNC_MAPPING
from mini_behavior.actions import Pickup, Drop, Toggle, Open, Close
from .objects import *
from .grid import BehaviorGrid, GridDimension, is_obj
from .window import Window
# from mini_behavior.window import Window
import random

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32


class MiniBehaviorEnv(MiniGridEnv):
    """
    2D grid world game environment
    """
    metadata = {
        # Deprecated: use 'render_modes' instead
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,  # Deprecated: use 'render_fps' instead
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        pickup = 3
        drop = 4
        drop_in = 5
        toggle = 6
        open = 7
        close = 8
        slice = 9
        cook = 10

    def __init__(
        self,
        mode='not_human',
        grid_size=None,
        width=None,
        height=None,
        num_objs=None,
        max_steps=1e5,
        see_through_walls=False,
        seed=1337,
        agent_view_size=7,
        highlight=True,
        tile_size=TILE_PIXELS,
    ):
        self.mode = mode
        self.last_action = None
        self.action_done = None

        self.render_dim = None

        self.highlight = highlight
        self.tile_size = tile_size

        # Initialize the RNG
        self.furniture_view = None

        if num_objs is None:
            num_objs = {}

        self.objs = {}
        self.obj_instances = {}

        for obj_type in num_objs.keys():
            self.objs[obj_type] = []
            for i in range(num_objs[obj_type]):
                obj_name = '{}_{}'.format(obj_type, i)

                if obj_type in OBJECT_CLASS.keys():
                    obj_instance = OBJECT_CLASS[obj_type](name=obj_name)
                else:
                    obj_instance = WorldObj(obj_type, None, obj_name)

                self.objs[obj_type].append(obj_instance)
                self.obj_instances[obj_name] = obj_instance

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(mission_space=mission_space,
                         grid_size=grid_size,
                         width=width,
                         height=height,
                         max_steps=max_steps,
                         see_through_walls=see_through_walls,
                         agent_view_size=agent_view_size,
                         )

        self.grid = BehaviorGrid(width, height)

        # Action enumeration for this environment, actions are discrete int
        self.actions = self.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.reset(seed)

    def set_render_mode(self, mode):
        self.render_mode = mode

    @staticmethod
    def _gen_mission():
        raise NotImplementedError

    def copy_objs(self):
        from copy import deepcopy
        return deepcopy(self.objs), deepcopy(self.obj_instances)

    # TODO: check this works
    def load_objs(self, state):
        obj_instances = state['obj_instances']
        grid = state['grid']
        for obj in self.obj_instances.values():
            if type(obj) != Wall and type(obj) != Door:
                load_obj = obj_instances[obj.name]
                obj.load(load_obj, grid, self)

        for obj in self.obj_instances.values():
            obj.contains = []
            for other_obj in self.obj_instances.values():
                if other_obj.check_rel_state(self, obj, 'inside'):
                    obj.contains.append(other_obj)

    # TODO: check this works
    def get_state(self):
        grid = self.grid.copy()
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        objs, obj_instances = self.copy_objs()
        state = {'grid': grid,
                 'agent_pos': agent_pos,
                 'agent_dir': agent_dir,
                 'objs': objs,
                 'obj_instances': obj_instances
                 }
        return state

    # Take in an object, return a list of [pos, dir]
    def get_obj_neighbor(self, obj):
        assert is_obj(obj) # Maybe this will fail for furniture? Not sure
        pos_x, pos_y = obj.cur_pos
        width = obj.width
        height = obj.height
        neighbor_list = []

        for x in range(pos_x, pos_x + width):
            neighbor_list.append([(x, pos_y - 1), 1]) # down
            neighbor_list.append([(x, pos_y + height), 3])  # up
        for y in range(pos_y, pos_y + height):
            neighbor_list.append([(pos_x - 1, y), 0])  # right
            neighbor_list.append([(pos_x + width, y), 2])  # left

        random.shuffle(neighbor_list)

        return neighbor_list

    # Take in an object, set agent to a position facing the object, return success
    def set_agent_to_neighbor(self, target_obj):
        # Check if holding the target obj
        if target_obj.cur_pos[0] == 0 and target_obj.cur_pos[1] == 0:
            return False

        neighbors = self.get_obj_neighbor(target_obj)
        for neighbor in neighbors:
            pos, dir = neighbor
            try:
                cur_test_cell = self.grid.get(*pos)
            except:
                import ipdb
                ipdb.set_trace()
            can_overlap = True
            for dim in cur_test_cell:
                for obj in dim:
                    if is_obj(obj) and not obj.can_overlap:
                        can_overlap = False
                        break
            if can_overlap:
                self.agent_dir = dir
                self.agent_pos = pos
                return True
        return False

    def save_state(self, out_file='cur_state.pkl'):
        state = self.get_state()
        with open(out_file, 'wb') as f:
            pkl.dump(state, f)
            print(f'saved to: {out_file}')

    # TODO: check this works
    def load_state(self, load_file):
        assert os.path.isfile(load_file)
        with open(load_file, 'rb') as f:
            state = pkl.load(f)
            self.load_objs(state)
            self.grid.load(state['grid'], self)
            self.agent_pos = state['agent_pos']
            self.agent_dir = state['agent_dir']
        return self.grid

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        self.carrying = set()

        for obj in self.obj_instances.values():
            obj.reset()

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # generate furniture view
        self.furniture_view = self.grid.render_furniture(tile_size=TILE_PIXELS, obj_instances=self.obj_instances)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        assert self.grid.is_empty(*self.agent_pos)

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs, {}

    def _gen_grid(self, width, height):
        self._gen_objs()
        assert self._init_conditions(), "Does not satisfy initial conditions"
        self.place_agent()

    def _gen_objs(self):
        raise NotImplementedError

    def _init_conditions(self):
        print('no init conditions')
        return True

    def _end_conditions(self):
        print('no end conditions')
        return False

    def generate_action(self):
        """
        Used for controlling scripted policy frequency
        """
        prob = 1.0
        if self.np_random.random() < prob:
            return self.hand_crafted_policy()
        else:
            return self.action_space.sample()

    # This function should only be called by RL classes with a stage reward
    def check_success(self):
        return self.stage_checkpoints["succeed"]

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            width = 1 if obj is None else obj.width
            height = 1 if obj is None else obj.height

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width - width + 1)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height - height + 1))
            ))

            valid = True

            for dx in range(width):
                for dy in range(height):
                    x = pos[0] + dx
                    y = pos[1] + dy

                    # Don't place the object on top of another object
                    if not self.grid.is_empty(x, y):
                        valid = False
                        break

                    # Don't place the object where the agent is
                    if np.array_equal((x, y), self.agent_pos):
                        valid = False
                        break

                    # Check if there is a filtering criterion
                    if reject_fn and reject_fn(self, (x, y)):
                        valid = False
                        break

            if not valid:
                continue

            break

        self.grid.set(*pos, obj)

        if obj:
            self.put_obj(obj, *pos)

        return pos

    def put_obj(self, obj, i, j, dim=0):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set(i, j, obj, dim)
        obj.init_pos = (i, j)
        obj.update_pos((i, j))

        if obj.is_furniture():
            for pos in obj.all_pos:
                self.grid.set(*pos, obj, dim)

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = BehaviorGrid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        if obs_grid.is_empty(vx, vy):
            return False

        for i in range(3):
            if [obj.type for obj in obs_cell[i]] != [obj.type for obj in world_cell[i]]:
                return False

        return True

    def step(self, action):
        # keep track of last action
        if self.mode == 'human':
            self.last_action = action
        else:
            self.last_action = self.actions(action)

        self.step_count += 1
        self.action_done = True

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
            else:
                self.action_done = False

        else:
            if self.mode == 'human':
                self.last_action = None
                if action == 'choose':
                    choices = self.all_reachable()
                    if not choices:
                        print("No reachable objects")
                    else:
                        # get all reachable objects
                        text = ''.join('{}) {} \n'.format(i, choices[i].name) for i in range(len(choices)))
                        obj = input("Choose one of the following reachable objects: \n{}".format(text))
                        obj = choices[int(obj)]
                        assert obj is not None, "No object chosen"

                        actions = []
                        for action in self.actions:
                            action_class = ACTION_FUNC_MAPPING.get(action.name, None)
                            if action_class and action_class(self).can(obj):
                                actions.append(action.name)

                        if len(actions) == 0:
                            print("No actions available")
                        else:
                            text = ''.join('{}) {} \n'.format(i, actions[i]) for i in range(len(actions)))

                            action = input("Choose one of the following actions: \n{}".format(text))
                            action = actions[int(action)] # action name

                            if action == 'drop' or action == 'drop_in':
                                dims = ACTION_FUNC_MAPPING[action](self).drop_dims(fwd_pos)
                                spots = ['bottom', 'middle', 'top']
                                text = ''.join(f'{dim}) {spots[dim]} \n' for dim in dims)
                                dim = input(f'Choose which dimension to drop the object: \n{text}')
                                ACTION_FUNC_MAPPING[action](self).do(obj, int(dim))
                            else:
                                ACTION_FUNC_MAPPING[action](self).do(obj) # perform action
                            self.last_action = self.actions[action]

                # Done action (not used by default)
                else:
                    assert False, "unknown action {}".format(action)
            else:
                # TODO: with agent centric, how does agent choose which obj to do the action on
                obj_action = self.actions(action).name.split('/') # list: [obj, action]

                # try to perform action
                obj = self.obj_instances[obj_action[0]]
                action_class = ACTION_FUNC_MAPPING[obj_action[1]]

                if action_class(self).can(obj):
                    action_class(self).do(obj)
                else:
                    self.action_done = False

        self.update_states()
        reward = self._reward()
        terminated = self._end_conditions()
        truncated = self.step_count >= self.max_steps
        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def all_reachable(self):
        return [obj for obj in self.obj_instances.values() if obj.check_abs_state(self, 'inreachofrobot')]

    def _reward(self):
        """
        This reward is only used by rl_env
        For other envs, might need to change this
        """
        stage_reward = self.update_stage_checkpoint()
        if self.stage_checkpoints["succeed"]:
            return 0
        if self.use_stage_reward:
            return stage_reward
        else:
            return -1

    def update_states(self):
        for obj in self.obj_instances.values():
            for name, state in obj.states.items():
                if state.type == 'absolute':
                    state._update(self)
        self.grid.state_values = {obj: obj.get_ability_values(self) for obj in self.obj_instances.values()}

    def render(self):
        """
        Render the whole-grid human view
        """
        mode = self.render_mode
        if mode == "human" and not self.window:
            self.window = Window("mini_behavior")
            self.window.show(block=False)

        # img = super().render(mode='rgb_array', highlight=highlight, tile_size=tile_size)
        self.render_mode = 'rgb_array'
        img = super().render()
        self.render_mode = mode

        if self.render_dim is None:
            img = self.render_furniture_states(img)
        else:
            img = self.render_furniture_states(img, dim=self.render_dim)

        if self.window:
            self.window.set_inventory(self)

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def render_states(self, tile_size=TILE_PIXELS):
        pos = self.front_pos
        imgs = []
        furniture = self.grid.get_furniture(*pos)
        img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
        if furniture:
            furniture.render(img)
            state_values = furniture.get_ability_values(self)
            GridDimension.render_furniture_states(img, state_values)
        imgs.append(img)

        for grid in self.grid.grid:
            furniture, obj = grid.get(*pos)
            state_values = obj.get_ability_values(self) if obj else None
            print(state_values)
            img = GridDimension.render_tile(furniture, obj, state_values, draw_grid_lines=False)
            imgs.append(img)

        return imgs

    def render_furniture_states(self, img, tile_size=TILE_PIXELS, dim=None):
        for obj in self.obj_instances.values():
            if obj.is_furniture():
                if dim is None or dim in obj.dims:
                    i, j = obj.cur_pos
                    ymin = j * tile_size
                    ymax = (j + obj.height) * tile_size
                    xmin = i * tile_size
                    xmax = (i + obj.width) * tile_size
                    sub_img = img[ymin:ymax, xmin:xmax, :]
                    state_values = obj.get_ability_values(self)
                    GridDimension.render_furniture_states(sub_img, state_values)
        return img

    def switch_dim(self, dim):
        self.render_dim = dim
        self.grid.render_dim = dim

    # This function make sure that the object is dropped in an available grid; to prevent object overlapping
    def drop_rand_dim(self, obj):
        if Drop(self).can(obj):
            drop_dim = obj.available_dims
            Drop(self).do(obj, np.random.choice(drop_dim))

    ################################
    #### For scripted policies  ####
    ################################
    def sample_nav_action(self):
        return self.np_random.choice([self.actions.left, self.actions.right, self.actions.forward])  # navigation

    def check_forward(self, cur):
        """
        helper function, check whether the agent can move forward
        """
        forward_grid = cur + DIR_TO_VEC[self.agent_dir]
        fwd_cell = self.grid.get(*forward_grid)
        return self.check_empty(fwd_cell), forward_grid

    def check_left(self, cur):
        """
        helper function, check whether the agent can move left
        """
        left_grid = cur + DIR_TO_VEC[(self.agent_dir - 1) % 4]
        left_cell = self.grid.get(*left_grid)
        return self.check_empty(left_cell), left_grid

    def check_right(self, cur):
        """
        helper function, check whether the agent can move right
        """
        right_grid = cur + DIR_TO_VEC[(self.agent_dir + 1) % 4]
        right_cell = self.grid.get(*right_grid)
        return self.check_empty(right_cell), right_grid

    def navigate_to(self, target):
        # Minigrid has a weird coordinate system, where the y axis is pointing down
        # There's a small bug with this navigation: if the target is in the opposite direction, it won't turn back until hitting an obstacle

        cur = np.array(self.agent_pos)
        dir = self.agent_dir

        diff = np.array(target) - cur
        can_forward, forward_grid = self.check_forward(cur)
        can_left, left_grid = self.check_left(cur)
        can_right, right_grid = self.check_right(cur)

        # When arrived, stop
        if np.all(forward_grid == target):
            return self.actions.forward
        if np.all(left_grid == target):
            return self.actions.left
        if np.all(right_grid == target):
            return self.actions.right

        def take_some_valid_action():
            if can_forward:
                return self.actions.forward
            elif not can_left:
                return self.actions.right
            elif not can_right:
                return self.actions.left
            else:
                return np.random.choice([self.actions.right, self.actions.left])

        # Facing right
        if dir == 0:
            if diff[0] > 0 and can_forward:
                return self.actions.forward
            elif diff[1] < 0 and can_left:
                return self.actions.left
            elif diff[1] > 0 and can_right:
                return self.actions.right
            # If the first three check fails, there is no clear action to take
            else:
                return take_some_valid_action()
        # Up
        elif dir == 1:
            if diff[1] > 0 and can_forward:
                return self.actions.forward
            elif diff[0] < 0 and can_right:
                return self.actions.right
            elif diff[0] > 0 and can_left:
                return self.actions.left
            # If the first three check fails, there is no clear action to take
            else:
                return take_some_valid_action()
        # Facing left
        elif dir == 2:
            if diff[0] < 0 and can_forward:
                return self.actions.forward
            elif diff[1] > 0 and can_left:
                return self.actions.left
            elif diff[1] < 0 and can_right:
                return self.actions.right
            # If the first three check fails, there is no clear action to take
            else:
                return take_some_valid_action()
        # Facing down
        elif dir == 3:
            if diff[1] < 0 and can_forward:
                return self.actions.forward
            elif diff[0] > 0 and can_right:
                return self.actions.right
            elif diff[0] < 0 and can_left:
                return self.actions.left
            # If the first three check fails, there is no clear action to take
            else:
                return take_some_valid_action()

    def go_drop(self, target, fwd_cell, dim, drop_action):
        if target in fwd_cell[dim]:
            action = drop_action
        else:
            action = self.navigate_to(target.cur_pos)
        return action

    def go_pickup(self, target, pickup_action):
        if Pickup(self).can(target):
            action = pickup_action
        else:
            action = self.navigate_to(target.cur_pos)
        return action

    def go_toggle(self, target, toggle_action):
        if Toggle(self).can(target):
            action = toggle_action
        else:
            action = self.navigate_to(target.cur_pos)
        return action

    def check_empty(self, cell):
        """
        A cell has 3 dimensions
        """
        for dim in cell:
            if not all(v is None for v in dim):
                return False
        return True

    ################################
    #### End of scripted policies  ####
    ################################

