import sys
from torch import nn as nn
from stable_baselines3.common.env_util import make_vec_env
from connectx.model_agent_loader import get_agent

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from connectx.lookahead import multistep_agent_factory
from connectx.look_agent_better import agent_factory
from connectx.environment import (
    ConnectFourGymV13,
    ConnectFourGymV2,
    ConnectFourGymV3,
    ConnectFourGymV4,
    ConnectFourGymV5,
    ConnectFourGymV6,
    ConnectFourGymV7,
    ConnectFourGymV8,
)
import importlib

if len(sys.argv) < 3:
    print("usage $ python src/train.py <model_name> <agent2> [<model_version>]")
    exit(1)

module = importlib.import_module(f"connectx.models.{sys.argv[1]}")

env_list = {
    "v3": ConnectFourGymV3,
    "v4": ConnectFourGymV4,
    "v5": ConnectFourGymV5,
    "v6": ConnectFourGymV6,
    "v7": ConnectFourGymV7,
    "v8": ConnectFourGymV8,
    "v13": ConnectFourGymV13,
    "default": ConnectFourGymV2,
}

selected_env = env_list["default"]
for e in env_list.items():
    if sys.argv[1].endswith(e[0]):
        selected_env = e[1]

if sys.argv[2].endswith("lookahead"):
    agent2 = multistep_agent_factory()
elif sys.argv[2].endswith("better"):
    agent2 = agent_factory()
elif sys.argv[2].endswith("negamax"):
    agent2 = "negamax"
elif sys.argv[2].endswith("random"):
    agent2 = "random"
else:

    try:
        model_agent_raw = sys.argv[2].split(":")
        model_agent = model_agent_raw[0]
        model_agent_version = int(model_agent_raw[1])


        agent2 = get_agent(model_agent, model_agent_version)
    except Exception as e:
        print(f"Error loading agent {e}")
        exit(1)    

env = selected_env(agent2=agent2)

n_envs = 1

# env = make_vec_env(lambda: env, n_envs=n_envs)

version = 0

if len(sys.argv) > 3:
    version = int(sys.argv[3])

print(f"Starting training from version {version}")
if version > 0:
    learner = module.get_model(env)
    learner.load_model_version(env, version)
else:
    learner = module.get_learner(env, new_model=True)

timesteps = 10e4

# Basic Training

epochs = None

try:

    i = 0
    while True:
        if epochs is not None and i >= epochs:
            print("Training Complete")
            break
        i = i + 1
        print(f"Model {learner.model_name} Version {version} (i {i}) training vs {agent2}")
        learner.learn(timesteps, n_envs=n_envs)
        version += timesteps
        version = int(version)
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

        update_model_data(learner.model_name, version, results_random, results_look, results_nega)

except KeyboardInterrupt:
    exit()