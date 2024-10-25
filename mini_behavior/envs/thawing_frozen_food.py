from mini_behavior.roomgrid import *
from mini_behavior.register import register
import numpy as np


class ThawingFrozenFoodEnv(RoomGrid):
    """
    Environment in which the agent needs to take objects out of frig and put them inside sink
    User can specify the type & amount of objects inside the scene through obj_in_scene
    """

    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            obj_in_scene={'olive': 1, 'fish': 1, 'date': 1},
            seed=42,
    ):

        num_objs = {'sink': 1, 'electric_refrigerator': 1}
        total_obj_num = 0
        for obj_key, value in obj_in_scene.items():
            num_objs[obj_key] = value
            total_obj_num += value

        self.obj_in_scene = obj_in_scene
        self.total_obj_num = total_obj_num
        self.mission = 'thaw frozen food'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         seed=seed,
                         )

    @staticmethod
    def _gen_mission():
        return "pick up food in the refrigerator and thaw them in the sink"

    def _gen_objs(self):

        electric_refrigerator = self.objs['electric_refrigerator'][0]
        sink = self.objs['sink'][0]

        frig_top = (2, 2)
        frig_size = (self.grid.width - 4 - electric_refrigerator.width, self.grid.height - 4 - electric_refrigerator.height)
        self.place_obj(electric_refrigerator, frig_top, frig_size)

        frig_poses = electric_refrigerator.all_pos

        def reject_fn(env, pos):
            """
            reject if frig next to sink
            """
            x, y = pos

            for mid in frig_poses:
                sx, sy = mid
                d = np.maximum(abs(sx - x), abs(sy - y))
                if d <= 1:
                    return True

            return False

        self.place_obj(sink, reject_fn=reject_fn)

        fridge_pos = self._rand_subset(electric_refrigerator.all_pos, self.total_obj_num)
        fridge_pos_idx = 0
        for obj_name, num in self.obj_in_scene.items():
            for idx in range(num):
                obj = self.objs[obj_name][idx]
                self.put_obj(obj, *fridge_pos[fridge_pos_idx], 1)
                obj.states['inside'].set_value(electric_refrigerator, True)
                fridge_pos_idx += 1

    def _end_conditions(self):
        for obj_name, num in self.obj_in_scene.items():
            for idx in range(num):
                obj = self.objs[obj_name][idx]
                if not obj.check_abs_state(self, "freezable") == 0:
                    return False
        return True

# non human input env
register(
    id='MiniGrid-ThawingFrozenFood-16x16-N2-v0',
    entry_point='mini_behavior.envs:ThawingFrozenFoodEnv'
)

# human input env
register(
    id='MiniGrid-ThawingFrozenFood-16x16-N2-v1',
    entry_point='mini_behavior.envs:ThawingFrozenFoodEnv',
    kwargs={'mode': 'human'}
)
