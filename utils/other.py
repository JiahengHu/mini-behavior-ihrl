import random
import numpy
import torch
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.reset()
    return env


def merge_dict(o_dict, m_list):
    for s_dict in m_list:
        for key in s_dict.keys():
            o_dict[key].append(s_dict[key])
    return o_dict