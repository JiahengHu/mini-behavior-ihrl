from mini_behavior.roomgrid import *
from mini_behavior.register import register
import numpy as np

class ThawingFrozenFoodEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='not_human',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'electric_refrigerator': 1, 'fish': 1, 'sink': 1}

        self.mission = 'thaw frozen food'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        fish = self.objs['fish']
        electric_refrigerator = self.objs['electric_refrigerator'][0]
        sink = self.objs['sink'][0]

        frig_top = (2, 2)
        frig_size = (self.grid.width - 4 - electric_refrigerator.width, self.grid.height - 4 - electric_refrigerator.height)
        self.place_obj(electric_refrigerator, frig_top, frig_size)

        frig_poses = electric_refrigerator.all_pos

        def reject_fn(env, pos):
            """
            reject if block the middle grid of the frig
            """
            x, y = pos

            for mid in frig_poses:
                sx, sy = mid
                d = np.maximum(abs(sx - x), abs(sy - y))
                if d <= 1:
                    return True

            return False

        self.place_obj(sink, reject_fn=reject_fn)

        fridge_pos = self._rand_subset(electric_refrigerator.all_pos, 1)
        # We make sure that all objects are of the same dimension
        self.put_obj(fish[0], *fridge_pos[0], 2)

        fish[0].states['inside'].set_value(electric_refrigerator, True)

    def _init_conditions(self):
        for obj in self.objs['fish']:
            assert obj.check_abs_state(self, 'freezable')

    def _end_conditions(self):
        fishes = self.objs['fish']
        sink = self.objs['sink'][0]

        for fish in fishes:
            if not fish.check_abs_state(self, "freezable") == 0:
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
