from mini_behavior.grid import is_obj
from mini_behavior.actions import Pickup, Drop, Toggle
from mini_behavior.floorplan import *

from enum import IntEnum
from gymnasium import spaces
from collections import OrderedDict
import math
from .installing_a_printer import InstallingAPrinterEnv


class FactoredInstallingAPrinterEnv(InstallingAPrinterEnv):
    """
    Environment in which the agent is instructed to install a printer
    This is a wrapper around the original mini-behavior environment where states are represented by category, and
    actions are converted to integer selection
    """
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        pickup = 3
        drop = 4
        toggle = 5
        clean = 6

    def __init__(
            self,
            mode='not_human',
            room_size=8,
            num_rows=1,
            num_cols=1,
            max_steps=200,
            use_stage_reward=False,
            seed=42,
            evaluate_graph=False,
            random_obj_pose=True,
            discrete_obs=True,
            task_name="install_printer",
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward
        self.evaluate_graph = evaluate_graph
        self.discrete_obs = discrete_obs
        self.task_name = task_name
        assert task_name in ["install_printer"]

        self.reward_range = (-math.inf, math.inf)
        self.random_obj_pose = random_obj_pose

        self.original_observation_space = spaces.Dict([
                ("agent", spaces.MultiDiscrete([self.room_size, self.room_size, 4])),
                ("printer", spaces.MultiDiscrete([self.room_size, self.room_size, 2, 2])),
                ("table", spaces.MultiDiscrete([self.room_size, self.room_size, 2]))
            ])

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         seed=seed
                         )

        if self.discrete_obs:
            self.observation_space = self.original_observation_space
        else:
            self.observation_space = spaces.Dict([(name, spaces.Box(low=-1, high=1, shape=[len(space.nvec)]))
                                                  for name, space in self.original_observation_space.spaces.items()])

        self.init_stage_checkpoint()
        self.desired_goal = None

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"printer_toggled": False, "printer_inhand": False, "succeed": False}
        self.stage_completion_tracker = 0

    def reset(self, seed=None, options=None):
        self.table_cleaned = False
        self.printer_ready = False

        obs, info = super().reset(seed=seed, options=options)
        self.init_stage_checkpoint()
        return obs, info

    def update_stage_checkpoint(self):
        if not self.stage_checkpoints["printer_toggled"]:
            if self.printer_toggledon:
                self.stage_checkpoints["printer_toggled"] = True
                self.stage_completion_tracker += 1
                return 1
        if not self.stage_checkpoints["printer_inhand"]:
            if self.printer_inhandofrobot:
                self.stage_checkpoints["printer_inhand"] = True
                self.stage_completion_tracker += 1
                return 1
        if not self.stage_checkpoints["succeed"]:
            if self._end_conditions():
                self.stage_checkpoints["succeed"] = True
                self.stage_completion_tracker += 1
                return 1
        return 0

    def hand_crafted_policy(self):
        """
        A hand-crafted function to select action for next step
        Notice that navigation is still random
        """
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if self.printer_ontop_table:
            if not self.printer_toggledon and Toggle(self).can(self.printer):
                action = self.actions.toggle  # toggle
            elif not self.table_cleaned:
                action = self.actions.clean
            else:
                action = self.action_space.sample()

        elif self.printer_inhandofrobot:
            if self.table in fwd_cell[1]:
                action = self.actions.drop # drop
            else:
                action = self.navigate_to(self.table.cur_pos)
        elif not self.printer_ontop_table and Pickup(self).can(self.printer):
            action = self.actions.pickup
        else:
            action = self.navigate_to(self.printer.cur_pos)

        return action

    def hand_crafted_lower_policy(self):
        factor = self.desired_goal[:3].argmax()
        parent = self.desired_goal[3:15].reshape(4, 3).argmax(axis=-1)
        action = np.random.randint(6)

        fwd_cell = self.grid.get(*self.front_pos)
        if factor == 0:
            if np.all(parent == np.array([1, 0, 0, 0])):
                action = np.random.randint(3, 6)
            elif np.all(parent == np.array([1, 0, 0, 1])):
                action = np.random.randint(2)
            elif np.all(parent == np.array([1, 1, 0, 0])):
                if Pickup(self).can(self.printer):
                    action = self.actions.forward
                else:
                    action = self.navigate_to(self.printer.cur_pos)
            elif np.all(parent == np.array([1, 0, 1, 0])):
                if self.table in fwd_cell[1]:
                    action = self.actions.forward
                else:
                    action = self.navigate_to(self.table.cur_pos)
        elif factor == 1:
            if np.all(parent == np.array([0, 1, 0, 0])):
                action = np.random.randint(6)
            elif np.all(parent == np.array([1, 1, 0, 1])):
                if Toggle(self).can(self.printer):
                    action = self.actions.toggle
                else:
                    action = self.navigate_to(self.printer.cur_pos)
            elif np.all(parent == np.array([1, 1, 1, 1])):
                if Toggle(self).can(self.printer) and self.printer_ontop_table:
                    action = self.actions.toggle  # toggle
                elif self.printer_inhandofrobot:
                    if self.table in fwd_cell[1]:
                        action = self.actions.drop # drop
                    else:
                        action = self.navigate_to(self.table.cur_pos)
                elif not self.printer_ontop_table and Pickup(self).can(self.printer):
                    action = self.actions.pickup
                else:
                    action = self.navigate_to(self.printer.cur_pos)

        return action

    def gen_obs(self):
        self.printer = self.objs['printer'][0]
        self.table = self.objs['table'][0]

        self.printer_inhandofrobot = int(self.printer.check_abs_state(self, 'inhandofrobot'))
        self.printer_ontop_table = int(self.printer.check_rel_state(self, self.table, 'onTop'))
        self.printer_toggledon = int(self.printer.check_abs_state(self, 'toggleable'))

        printer_pos = self.agent_pos if self.printer_inhandofrobot else self.printer.cur_pos

        obs = {"agent": np.array([*self.agent_pos, self.agent_dir]),
               "printer": np.array([*printer_pos, self.printer_toggledon, self.printer_ready]),
               "table": np.array([*self.table.cur_pos, self.table_cleaned])}
        if not self.discrete_obs:
            for k, v in obs.items():
                obs[k] = (2. * v / (self.original_observation_space[k].nvec - 1) - 1).astype(np.float32)

        return obs

    def _gen_objs(self):
        if self.random_obj_pose:
            return super()._gen_objs()
        else:
            printer = self.objs['printer'][0]
            table = self.objs['table'][0]

            table_pos = (1, 2)
            printer_pos = (6, 5)
            self.put_obj(table, *table_pos, 0)
            self.put_obj(printer, *printer_pos, 0)

    def place_agent(self):
        if self.random_obj_pose:
            return super().place_agent()
        else:
            self.agent_pos = np.array([4,4])
            self.agent_dir = 0
            return self.agent_pos

    def step(self, action):
        # print("action", self.actions(action).name)
        self.update_states()

        if self.desired_goal is not None:
            action = self.hand_crafted_lower_policy()

        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        table_infront = self.table in fwd_cell[1]

        picked = dropped = toggled = cleaned = printer_got_ready = False

        if self.printer_toggledon and self.printer_ontop_table and self.table_cleaned and not self.printer_ready:
            self.printer_ready = True
            printer_got_ready = True

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
                        obstacle = obj
                        break

            if can_overlap:
                self.agent_pos = fwd_pos
        elif action == self.actions.pickup:
            if Pickup(self).can(self.printer):
                Pickup(self).do(self.printer)
                picked = True
        elif action == self.actions.drop:
            if Drop(self).can(self.printer):
                Drop(self).do(self.printer, 2)
                dropped = True
        elif action == self.actions.toggle:
            if Toggle(self).can(self.printer):
                Toggle(self).do(self.printer)
                toggled = True
        elif action == self.actions.clean:
            if table_infront and not self.table_cleaned:
                self.table_cleaned = True
                cleaned = True

        info = {"success": self.check_success()}

        # We need to evaluate mask before we call "gen_obs"
        if self.evaluate_graph:
            feature_dim = 10
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            printer_pos_idxes = slice(3, 5)
            printer_state_idx = 5
            printer_ready_idx = 6
            table_pos_idxes = slice(7, 9)
            table_clean_idx = 9
            action_idx = 10

            # Rotate left
            if action == self.actions.left or action == self.actions.right:
                mask[agent_dir_idx, action_idx] = True

            # Move forward
            elif action == self.actions.forward:
                pos_idx = self.agent_dir % 2
                printer_pos_idx = pos_idx + 3
                if can_overlap:
                    mask[pos_idx, agent_dir_idx] = True
                    mask[pos_idx, action_idx] = True
                    if self.printer_inhandofrobot:
                        mask[printer_pos_idx, agent_pos_idxes] = True
                        mask[printer_pos_idx, agent_dir_idx] = True
                        mask[printer_pos_idx, printer_pos_idxes] = True
                        mask[printer_pos_idx, action_idx] = True
                else:
                    mask[pos_idx, agent_pos_idxes] = True
                    mask[pos_idx, agent_dir_idx] = True
                    if obstacle == self.printer:
                        mask[pos_idx, printer_pos_idxes] = True
                    elif obstacle == self.table:
                        mask[pos_idx, table_pos_idxes] = True
                    if self.printer_inhandofrobot:
                        mask[printer_pos_idx, agent_pos_idxes] = True
                        mask[printer_pos_idx, agent_dir_idx] = True
                        mask[printer_pos_idx, printer_pos_idxes] = True
                        if obstacle == self.table:
                            mask[printer_pos_idx, table_pos_idxes] = True
                            mask[printer_pos_idx, action_idx] = True
            elif action == self.actions.pickup:
                if picked:
                    mask[printer_pos_idxes, agent_pos_idxes] = True
                    mask[printer_pos_idxes, agent_dir_idx] = True
                    mask[printer_pos_idxes, printer_pos_idxes] = True
                    mask[printer_pos_idxes, action_idx] = True
            elif action == self.actions.drop:
                if dropped:
                    mask[printer_pos_idxes, agent_pos_idxes] = True
                    mask[printer_pos_idxes, agent_dir_idx] = True
                    mask[printer_pos_idxes, printer_pos_idxes] = True
                    mask[printer_pos_idxes, action_idx] = True
            elif action == self.actions.toggle:
                if toggled:
                    mask[printer_state_idx, agent_pos_idxes] = True
                    mask[printer_state_idx, agent_dir_idx] = True
                    mask[printer_state_idx, printer_pos_idxes] = True
                    mask[printer_state_idx, action_idx] = True
            elif action == self.actions.clean:
                if cleaned:
                    mask[table_clean_idx, agent_pos_idxes] = True
                    mask[table_clean_idx, agent_dir_idx] = True
                    mask[table_clean_idx, table_pos_idxes] = True
                    mask[table_clean_idx, action_idx] = True

            # Add causal mask for printer_ready
            if printer_got_ready:
                mask[printer_ready_idx, printer_pos_idxes] = True
                mask[printer_ready_idx, printer_state_idx] = True
                mask[printer_ready_idx, table_pos_idxes] = True
                mask[printer_ready_idx, table_clean_idx] = True

            info["variable_graph"] = mask

            num_factors = 3
            agent_idxes = slice(0, 3)
            printer_idxes = slice(3, 7)
            table_idxes = slice(7, 10)
            factor_mask = np.zeros((num_factors, num_factors + 1), dtype=bool)
            for i, idxes in enumerate([agent_idxes, printer_idxes, table_idxes]):
                for j, pa_idxes in enumerate([agent_idxes, printer_idxes, table_idxes, action_idx]):
                    factor_mask[i, j] = mask[idxes, pa_idxes].any()
            info["factor_graph"] = factor_mask

        obs = self.gen_obs()
        reward = self._reward()

        terminated = False  # self._end_conditions()
        truncated = self.step_count >= self.max_steps

        info["stage_completion"] = self.stage_completion_tracker

        return obs, reward, terminated, truncated, info


register(
    id='MiniGrid-installing_printer-v0',
    entry_point='mini_behavior.envs:FactoredInstallingAPrinterEnv',
    kwargs={}
)
