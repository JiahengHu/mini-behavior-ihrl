import argparse
import numpy
import mini_behavior

from utils.penv import ParallelEnv
from utils.other import merge_dict, make_env, seed, device
import pickle
from collections import defaultdict
import time


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-SimpleInstallingAPrinter-8x8-N2-v0', # 'MiniGrid-FloorPlanEnv-16x16-N1-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=8,
                    help="number of parallel envs to run")
parser.add_argument("--env_steps", type=int, default=100000,
                    help="number of steps to run")
parser.add_argument("--save_freq", type=int, default=1000,
                    help="number of steps before save")
parser.add_argument("--visualize", default=False, action="store_true",
                    help="Visualize the agent vehavior")
args = parser.parse_args()

# Set seed for all randomness sources
seed(args.seed)
procs = args.procs
num_of_data = args.env_steps

# Set device
print(f"Device: {device}\n")
print(f"Number of Envs: {procs}\n")
print(f"Env Name: {args.env}\n")
print(f"Total number of collected data: {num_of_data * procs}\n")

envs = []
for i in range(procs):
    envs.append(make_env(args.env, args.seed + 10000 * i))
env = ParallelEnv(envs)

print("Environment loaded\n")

# Load agent
actions_list = []
done_list = []
state_dict = defaultdict(list)

state = env.reset()
state_dict = merge_dict(state_dict, state)
done_list += [False]*procs
action = [envs[i].generate_action() for i in range(procs)]
actions_list += action

for i in range(num_of_data):
    # Do one agent-environment interaction
    state, reward, done, _ = env.step(action)

    # Store info
    state_dict = merge_dict(state_dict, state)
    done_list += done
    action = env.generate_action()
    actions_list += action


    # Feel free to change how you want to store the data
    if (i+1) % args.save_freq == 0:
        print(f"saving iteration {i+1}...")
        with open("minigrid_causal_data", "wb") as fp:  # Pickling
            pickle.dump([state_dict, actions_list, done_list], fp)

        # # stats for done portion
        # print(f"Portion of dones: {sum(done_list)/ len(done_list)}")


