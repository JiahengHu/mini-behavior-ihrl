# Mini-BEHAVIOR
###  MiniGrid Implementation of BEHAVIOR Tasks

Adapted from https://github.com/StanfordVL/mini_behavior

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
Use the gynamsium and minigrid version listed in setup.py.

### Run Code 
To run in interactive mode: ./manual_control.py


### References
```
@article{jin2023minibehavior,
      title={Mini-BEHAVIOR: A Procedurally Generated Benchmark for Long-horizon Decision-Making in Embodied AI}, 
      author={Emily Jin and Jiaheng Hu and Zhuoyi Huang and Ruohan Zhang and Jiajun Wu and Li Fei-Fei and Roberto Mart{\'i}n-Mart{\'i}n},
      year={2023},
      journal={arXiv preprint 2310.01824},
}
```

