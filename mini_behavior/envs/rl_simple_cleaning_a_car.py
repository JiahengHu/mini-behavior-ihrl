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
from .cleaning_a_car import CleaningACarEnv


class SimpleCleaningACarEnv(CleaningACarEnv):
    """
    Environment in which the agent is instructed to clean a car
    This is a wrapper around the original mini-behavior environment where:
    - states are represented by category, and
    - actions are converted to integer selection
    In order to make it work better with causal influence detection, we remove the navigation action and create the following alternatives:
    - MoveTo (Rag, Soap, Bucket, Sink, Car, empty)
    - Pickup (rag, soap)
    - Drop (rag, soap)
    - Toggle (sink)
    """
    class Actions(IntEnum):
        # left = 0
        # right = 1
        # forward = 2
        move_to_rag = 0
        move_to_soap = 1
        move_to_bucket = 2
        pickup_soap = 3
        drop_soap = 4
        pickup_rag = 5
        drop_rag = 6
        toggle_sink = 7
        move_to_car = 8
        move_to_sink = 9
        switch_tv = 10
        move_to_tv = 11


    def __init__(
            self,
            mode='not_human',
            room_size=10,
            num_rows=1,
            num_cols=1,
            max_steps=300,
            use_stage_reward=False,
            add_noisy_tv=True,
    ):
        self.room_size = room_size
        self.use_stage_reward = use_stage_reward
        self.tv_dim = 10
        self.tv_color = 10

        super().__init__(mode=mode,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         add_noisy_tv=add_noisy_tv,
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
        self.stage_checkpoints = {"rag_pickup":False, "rag_soaked": False, "car_not_stain": False, "soap_in_bucket": False, "succeed": False}
        self.stage_completion_tracker = 0

    def reset(self):
        self.tv_state = np.zeros(self.tv_dim)
        obs = super().reset()
        self.init_stage_checkpoint()
        return obs

    def observation_dims(self):
        obs_dim = {
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
        if self.add_noisy_tv:
            obs_dim["tv_state"] = np.array([self.tv_color] * self.tv_dim)
            obs_dim["tv_pos"] = np.array([self.room_size, self.room_size])
        return obs_dim

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

        if self.car_stain:
            if self.rag.check_abs_state(self, 'inhandofrobot'):
                if self.rag_soak != 5:
                    if self.sink in fwd_cell[0]:
                        action = self.actions.drop_rag
                    else:
                        action = self.actions.move_to_sink
                else:
                    if self.car in fwd_cell[0]:
                        action = self.actions.drop_rag
                    else:
                        action = self.actions.move_to_car
            elif self.rag.check_rel_state(self, self.sink, "atsamelocation"):
                if not self.sink_toggled:
                    if Toggle(self).can(self.sink):
                        action = self.actions.toggle_sink
                    else:
                        action = self.actions.move_to_sink
                elif self.rag_soak != 5:
                    action = self.actions.move_to_sink
                else:
                    if Pickup(self).can(self.rag):
                        action = self.actions.pickup_rag
                    else:
                        action = self.actions.move_to_rag
            else:
                if Pickup(self).can(self.rag):
                    action = self.actions.pickup_rag
                else:
                    action = self.actions.move_to_rag
        else:
            rag_in_bucket = self.rag.check_rel_state(self, self.bucket, "atsamelocation")
            soap_in_bucket = self.soap.check_rel_state(self, self.bucket, "atsamelocation")
            if rag_in_bucket and soap_in_bucket:
                action = self.actions.move_to_rag
            elif not soap_in_bucket:
                if self.soap.check_abs_state(self, 'inhandofrobot'):
                    if self.bucket in fwd_cell[0]:
                        action = self.actions.drop_soap
                    else:
                        action = self.actions.move_to_bucket
                else:
                    if Pickup(self).can(self.soap):
                        action = self.actions.pickup_soap
                    else:
                        action = self.actions.move_to_soap
            elif not rag_in_bucket:
                if self.rag.check_abs_state(self, 'inhandofrobot'):
                    if self.bucket in fwd_cell[0]:
                        action = self.actions.drop_rag
                    else:
                        action = self.actions.move_to_bucket
                else:
                    if Pickup(self).can(self.rag):
                        action = self.actions.pickup_rag
                    else:
                        action = self.actions.move_to_rag
            else:
                print("reaching here is impossible")
                raise NotImplementedError

        return action

    def update_stage_checkpoint(self):
        self.stage_completion_tracker += 1
        if not self.stage_checkpoints["rag_pickup"]:
            if self.rag.check_abs_state(self, 'inhandofrobot'):
                self.stage_checkpoints["rag_pickup"] = True
                return 1
        if not self.stage_checkpoints["rag_soaked"]:
            if self.rag_soak == 5:
                self.stage_checkpoints["rag_soaked"] = True
                return 1
        if not self.stage_checkpoints["car_not_stain"]:
            if not self.car_stain:
                self.stage_checkpoints["car_not_stain"] = True
                return 1
        if not self.stage_checkpoints["soap_in_bucket"]:
            if self.soap.check_rel_state(self, self.bucket, 'atsamelocation'):
                self.stage_checkpoints["soap_in_bucket"] = True
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

        # these states are important for figuring out the GT mask (used for next timestep)
        self.rag_on_car = self.car.check_rel_state(self, self.rag, 'atsamelocation')
        self.soap_in_bucket = self.bucket.check_rel_state(self, self.soap, 'atsamelocation')
        self.rag_in_bucket = self.bucket.check_rel_state(self, self.rag, 'atsamelocation')
        self.rag_in_sink = self.sink.check_rel_state(self, self.rag, 'atsamelocation')


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

        if self.add_noisy_tv:
            self.tv = self.objs['tv'][0]
            obs["tv_state"] = self.tv_state
            obs["tv_pos"] = np.array(self.tv.cur_pos)

        return obs

    def test_seed(self, seed=None):
        # Just to test if this function is working as intended
        print(f"{seed} test seed result")

    def step(self, action, evaluate_mask=True):
        self.update_states()
        self.step_count += 1
        # Get the position and contents in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        move_success = picked_rag = picked_soap = toggled = switched_tv = False

        if action == self.actions.move_to_sink:
            move_success = self.set_agent_to_neighbor(self.sink)
        elif action == self.actions.move_to_rag:
            move_success = self.set_agent_to_neighbor(self.rag)
        elif action == self.actions.move_to_bucket:
            move_success = self.set_agent_to_neighbor(self.bucket)
        elif action == self.actions.move_to_car:
            move_success = self.set_agent_to_neighbor(self.car)
        elif action == self.actions.move_to_soap:
            move_success = self.set_agent_to_neighbor(self.soap)
        elif action == self.actions.move_to_tv:
            move_success = False
            if self.add_noisy_tv:
                move_success = self.set_agent_to_neighbor(self.tv)

        elif action == self.actions.pickup_rag:
            if Pickup(self).can(self.rag):
                Pickup(self).do(self.rag)
                picked_rag = True
        elif action == self.actions.pickup_soap:
            if Pickup(self).can(self.soap):
                Pickup(self).do(self.soap)
                picked_soap = True
        elif action == self.actions.toggle_sink:
            if Toggle(self).can(self.sink):
                Toggle(self).do(self.sink)
                toggled = True
        elif action == self.actions.drop_rag:
            self.drop_rand_dim(self.rag)
        elif action == self.actions.drop_soap:
            self.drop_rand_dim(self.soap)
        elif action == self.actions.switch_tv:
            if self.add_noisy_tv:
                if self.tv in fwd_cell[0]:
                    self.tv_state = np.random.randint(self.tv_color, size=self.tv_dim)
                    switched_tv = True
        else:
            print(action)
            raise NotImplementedError

        # We need to evaluate mask before we call "gen_obs"
        if evaluate_mask:
            if self.add_noisy_tv:
                feature_dim = 29  # 10 dims for tv, but this is not ideal
                tv_state_idx = slice(17, 27)
                tv_pos_idx = slice(27, 29)
            else:
                feature_dim = 17  # action, car_pos, bucket_pos, agent_pos, agent_dir, soap_pos
            mask = np.eye(feature_dim, feature_dim + 1, dtype=bool)
            agent_pos_idxes = slice(0, 2)
            agent_dir_idx = 2
            rag_pos_idxes = slice(3, 5)
            rag_soak_idx = 5
            rag_clean_idx = 6
            soap_pos_idxes = slice(7, 9)
            sink_pos_idxes = slice(9, 11)
            sink_state_idx = 11
            car_pos_idxes = slice(12, 14)
            car_state_idx = 14
            bucket_pos_idxes = slice(15, 17)
            action_idx = -1

            if action in [self.actions.move_to_sink, self.actions.move_to_soap, self.actions.move_to_car,
                           self.actions.move_to_bucket, self.actions.move_to_rag]:
                if move_success:
                    mask[agent_dir_idx, action_idx] = True
                    mask[agent_pos_idxes, action_idx] = True

            elif action == self.actions.pickup_soap:
                if picked_soap:
                    mask[soap_pos_idxes, agent_pos_idxes] = True
                    mask[soap_pos_idxes, agent_dir_idx] = True
                    mask[soap_pos_idxes, soap_pos_idxes] = True
                    mask[soap_pos_idxes, action_idx] = True

            elif action == self.actions.pickup_rag:
                if picked_rag:
                    mask[rag_pos_idxes, agent_pos_idxes] = True
                    mask[rag_pos_idxes, agent_dir_idx] = True
                    mask[rag_pos_idxes, rag_pos_idxes] = True
                    mask[rag_pos_idxes, action_idx] = True

            elif action == self.actions.drop_rag:
                mask[rag_pos_idxes, agent_pos_idxes] = True
                mask[rag_pos_idxes, agent_dir_idx] = True
                mask[rag_pos_idxes, action_idx] = True

            elif action == self.actions.drop_soap:
                mask[soap_pos_idxes, agent_pos_idxes] = True
                mask[soap_pos_idxes, agent_dir_idx] = True
                mask[soap_pos_idxes, action_idx] = True

            elif action == self.actions.toggle_sink:
                if toggled:
                    mask[sink_state_idx, action_idx] = True
                    mask[sink_state_idx, agent_pos_idxes] = True
                    mask[sink_state_idx, agent_dir_idx] = True
                    mask[sink_state_idx, sink_pos_idxes] = True

            if self.rag_on_car and self.rag_soak and self.car_stain:
                mask[car_state_idx, rag_pos_idxes] = True
                mask[car_state_idx, car_pos_idxes] = True
                mask[car_state_idx, rag_soak_idx] = True
                mask[rag_clean_idx, rag_pos_idxes] = True
                mask[rag_clean_idx, car_pos_idxes] = True
                mask[rag_clean_idx, rag_soak_idx] = True

            if self.rag_in_bucket and self.soap_in_bucket and self.rag_cleanness > 0:
                mask[rag_clean_idx, rag_pos_idxes] = True
                mask[rag_clean_idx, soap_pos_idxes] = True
                mask[rag_clean_idx, bucket_pos_idxes] = True

            if self.rag_in_sink and self.sink_toggled and self.rag_soak < 5:
                mask[rag_soak_idx, rag_pos_idxes] = True
                mask[rag_soak_idx, sink_pos_idxes] = True
                mask[rag_soak_idx, sink_state_idx] = True

            if switched_tv:
                mask[tv_state_idx, action_idx] = True
                mask[tv_state_idx, agent_pos_idxes] = True
                mask[tv_state_idx, agent_dir_idx] = True
                mask[tv_state_idx, tv_pos_idx] = True

        reward = self._reward()
        # done = self._end_conditions() or self.step_count >= self.max_steps
        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        info = {"success": self.check_success(), "stage_completion": self.stage_completion_tracker}

        if evaluate_mask:
            info["local_causality"] = mask

        return obs, reward, done, info


register(
    id='MiniGrid-clearning_car-v0',
    entry_point='mini_behavior.envs:SimpleCleaningACarEnv'
)
