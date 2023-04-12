# FROM RL TORCH

import argparse
import numpy
import mini_behavior

from rl.utils.env import make_env
from rl.utils.other import seed, device

# Parse arguments

# env = 'MiniGrid-ThrowLeftoversMulti-16x16-N2-v0'
# model = 'throw_leftovers_model'
# env = 'MiniGrid-ThrowLeftoversNavigation-8x8-N2-v0'
# model = 'MiniGrid-ThrowLeftoversNavigation-8x8-N2-v0_22-07-20-16-13-11'

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-SimpleInstallingAPrinter-8x8-N2-v0', # 'MiniGrid-FloorPlanEnv-16x16-N1-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--seed", type=int, default=20,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.seed 0_2)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = env.generate_action()  # action_space.sample()
        obs, reward, done, _ = env.step(action)
        # print(env.last_action.name)
        print(action)
        print(obs)

        if done or env.window.closed:
            print("episode done")
            break

    print("one episode done \n")
    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
