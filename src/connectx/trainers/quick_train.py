import sys
from torch import nn as nn

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from connectx.lookahead import multistep_agent_factory

from connectx.environment import ConnectFourGymV2, ConnectFourGymV3, ConnectFourGymV4, ConnectFourGymV5, ConnectFourGymV6


import importlib


if len(sys.argv) < 2:
    print("usage $ python src/train.py <model_name>")
    exit(1)

module = importlib.import_module(f"connectx.models.{sys.argv[1]}")

if sys.argv[1].endswith("v3"):
    print("Using V3")
    env = ConnectFourGymV3(agent2="random")
elif sys.argv[1].endswith("v4"):
    print("Using V4")
    env = ConnectFourGymV4(agent2="random")
elif sys.argv[1].endswith("v5"):
    print("Using V5")
    env = ConnectFourGymV5(agent2="random")
elif sys.argv[1].endswith("v6"):
    print("Using V5")
    env = ConnectFourGymV6(agent2="random")
else:
    print("Using V2")
    env = ConnectFourGymV2(agent2="random")

learner = module.get_learner(env)

timesteps = 10e3

# Basic Training

iterations = 30
version = 0
for i in range(1, iterations + 1):
    version = int(timesteps * i)
    print(f"Model {learner.model_name} Version {version} (i {i}) training vs Random")
    learner.learn(timesteps)

    learner.save(version)
    agent = learner.get_agent()
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)
    print("Vs Lookahead:")

    results_look = get_win_percentages(agent, multistep_agent_factory())
    print_win_percentages(results_look)

    update_model_data(learner.model_name, version, results_random, results_look)


print("Training Done")
