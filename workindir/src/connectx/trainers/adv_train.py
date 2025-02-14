import sys
from torch import nn as nn

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from stable_baselines3.common.env_util import make_vec_env
from connectx.lookahead import multistep_agent_factory
from connectx.environment import (
    ConnectFourGymV10,
    ConnectFourGymV2,
    ConnectFourGymV3,
    ConnectFourGymV4,
    ConnectFourGymV5,
    ConnectFourGymV6,
    ConnectFourGymV7,
    ConnectFourGymV8,
    ConnectFourGymV9,
)


import importlib


# if len(sys.argv) < 3:
if len(sys.argv) < 2:
    print("usage $ python src/train.py <model_name>")
    # print("usage $ python src/train.py <model_name> <model_version>")
    exit(1)

module = importlib.import_module(f"connectx.models.{sys.argv[1]}")

env_list = {
    "v3": ConnectFourGymV3,
    "v4": ConnectFourGymV4,
    "v5": ConnectFourGymV5,
    "v6": ConnectFourGymV6,
    "v7": ConnectFourGymV7,
    "v8": ConnectFourGymV8,
    "v9": ConnectFourGymV9,
    "v10": ConnectFourGymV10,
    "default": ConnectFourGymV2,
}

selected_env = env_list["default"]
for e in env_list.items():
    if sys.argv[1].endswith(e[0]):
        selected_env = e[1]

env = selected_env(agent2=multistep_agent_factory())
# env = make_vec_env(lambda: env, n_envs=4)
print("[+] Selected env", selected_env)

learner = module.get_learner(env)

timesteps = 10e4

# Basic Training

epochs = None

# version = int(sys.argv[2])
version = 0
# learner.load_model_version(env, version)
i = 0
while True:
    if epochs is not None and i >= epochs:
        print("Training Complete")
        break
    i = i + 1
    version += timesteps
    version = int(version)
    print(f"Model {learner.model_name} Version {version} (i {i}) training vs Lookout")
    learner.learn(timesteps)

    learner.save(version)
    agent = learner.get_agent()
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)

    print("Vs Lookahead:")
    results_look = get_win_percentages(agent, multistep_agent_factory())
    print_win_percentages(results_look)

    print("Vs negamax:")
    results_nega = get_win_percentages(agent, "negamax")
    print_win_percentages(results_nega)

    update_model_data(learner.model_name, version, results_random, results_look, results_negamax=results_nega)
