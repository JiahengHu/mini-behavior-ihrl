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

    def __init__(
            self,
            mode='not_human',
            room_size=8,
            num_rows=1,
            num_cols=1,
            max_steps=200,
            use_stage_reward=False,
            seed=42,
            evaluate_mask=False
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward
        self.evaluate_mask = evaluate_mask

        self.reward_range = (-math.inf, math.inf)

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         seed=seed
                         )

        self.observation_space = spaces.Dict([
            ("agent", spaces.MultiDiscrete([self.room_size, self.room_size, 4])),
            ("printer", spaces.MultiDiscrete([self.room_size, self.room_size, 2, 2])),
            ("table", spaces.MultiDiscrete([self.room_size, self.room_size]))
        ])

        self.init_stage_checkpoint()

    def init_stage_checkpoint(self):
        """
        These values are used for keeping track of partial completion reward
        """
        self.stage_checkpoints = {"printer_toggled": False, "printer_inhand": False, "succeed": False}
        self.stage_completion_tracker = 0

    def reset(self, seed=None, options=None):
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

        if not self.printer_toggledon and Toggle(self).can(self.printer) and self.printer_ontop_table:
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

        obs = {"agent": np.array([*self.agent_pos, self.agent_dir]),
               "printer": np.array([*self.printer.cur_pos, self.printer_toggledon, self.printer_ontop_table]),
               "table": np.array(self.table.cur_pos)}

        return obs

    def step(self, action):
        self.update_states()

        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        picked = dropped = toggled = False

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
                # Modified Env: can only toggle if one table
                if self.printer_ontop_table:
                    Toggle(self).do(self.printer)
                    toggled = True

        info = {"success": self.check_success()}

        # We need to evaluate mask before we call "gen_obs"
        if self.evaluate_mask:
            feature_dim = 9
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            printer_pos_idxes = slice(3, 5)
            printer_state_idx = 5
            table_pos_idxes = slice(6, 8)
            printer_table_idx = 8
            action_idx = 9

            # Rotate left
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
                    if obj == self.printer:
                        mask[pos_idx, printer_pos_idxes] = True
                    elif obj == self.table:
                        mask[pos_idx, table_pos_idxes] = True
            elif action == self.actions.pickup:
                if picked:
                    mask[printer_pos_idxes, agent_pos_idxes] = True
                    mask[printer_pos_idxes, agent_dir_idx] = True
                    mask[printer_pos_idxes, printer_pos_idxes] = True
                    mask[printer_pos_idxes, action_idx] = True
            elif action == self.actions.drop:
                mask[printer_pos_idxes, agent_pos_idxes] = True
                mask[printer_pos_idxes, agent_dir_idx] = True
                if dropped:
                    mask[printer_pos_idxes, action_idx] = True
            elif action == self.actions.toggle:
                if toggled:
                    mask[printer_state_idx, action_idx] = True
                    # if not self.printer_inhandofrobot:
                    mask[printer_state_idx, agent_pos_idxes] = True
                    mask[printer_state_idx, agent_dir_idx] = True
                    mask[printer_state_idx, printer_pos_idxes] = True

            # Add causal mask for printer_on_table
            mask[printer_table_idx, printer_table_idx] = False
            if self.printer_inhandofrobot and self.table in fwd_cell[1] and action == self.actions.drop:
                mask[printer_table_idx, table_pos_idxes] = True
                mask[printer_table_idx, agent_pos_idxes] = True
                mask[printer_table_idx, agent_dir_idx] = True
                mask[printer_table_idx, action_idx] = True
            elif self.printer_ontop_table:
                mask[printer_table_idx, printer_pos_idxes] = True
                mask[printer_table_idx, table_pos_idxes] = True
                if action == self.actions.pickup and picked:
                    mask[printer_table_idx, agent_pos_idxes] = True
                    mask[printer_table_idx, agent_dir_idx] = True
                    mask[printer_table_idx, action_idx] = True

            info["true_graph"] = mask

        obs = self.gen_obs()
        reward = self._reward()

        terminated = self._end_conditions()
        truncated = self.step_count >= self.max_steps

        info["stage_completion"] = self.stage_completion_tracker

        return obs, reward, terminated, truncated, info


register(
    id='MiniGrid-installing_printer-v0',
    entry_point='mini_behavior.envs:FactoredInstallingAPrinterEnv',
    kwargs={}
)
