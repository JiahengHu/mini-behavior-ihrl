# Mini-BEHAVIOR
###  MiniGrid Implementation of BEHAVIOR Tasks

### For Local Causality
**Current tested environments:**
1. MiniGrid-installing_printer-v0

* To use the environment:
```python
import gymnasium as gym
import mini_behavior
from mini_behavior.wrappers.flatten_dict_observation import FlattenDictObservation

kwargs = {"room_size": 10,
          "num_rows": 1,
          "num_cols": 1,
          "max_steps": 200,
          "evaluate_graph": True}
env = gym.make("MiniGrid-installing_printer-v0", **kwargs)
env = FlattenDictObservation(env)
```

### Environment Setup
```buildoutcfg
pip install -e .
```
In the setup.py, I list the version gynamsium and minigrid I am using, older version may also work.

### Run Code 
To run in interactive mode: ./manual_control.py

### Floor plan to Mini-Behavior Environment
* add image file of floor plan to gym_minigrid/scenes directory
* run script to process floor plan and save grid to gym_minigrid/grids directory: `python convert_scenes.py --imgs IMG_FILENAMES`
* `floorplan.py` will register the floor plan of each `IMG_FILENAME` as an environment with:
    * `id='MiniGrid-IMG_FILENAME-0x0-N1-v0'`
    * `entry_point='gym_minigrid.envs:FloorPlanEnv'`
