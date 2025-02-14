import sys
from torch import nn as nn

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from connectx.lookahead import multistep_agent_factory
from connectx.environment import (
    ConnectFourGymV2,
    ConnectFourGymV3,
    ConnectFourGymV4,
    ConnectFourGymV5,
    ConnectFourGymV6,
    ConnectFourGymV7,
    ConnectFourGymV8,
)


import random
import importlib


if len(sys.argv) < 2:
    print("usage $ python src/train.py <model_name> [<version>]")
    exit(1)

module = importlib.import_module(f"connectx.models.{sys.argv[1]}")


def get_env(agent2):

    env_list = {
        "v3": ConnectFourGymV3,
        "v4": ConnectFourGymV4,
        "v5": ConnectFourGymV5,
        "v6": ConnectFourGymV6,
        "v7": ConnectFourGymV7,
        "v8": ConnectFourGymV8,
        "default": ConnectFourGymV2,
    }

    selected_env = env_list["default"]
    for e in env_list.items():
        if sys.argv[1].endswith(e[0]):
            selected_env = e[1]

    env = selected_env(agent2=agent2)

    return env


env = get_env(multistep_agent_factory())


learner = module.get_learner(env)

timesteps = 10e5

# Basic Training

epochs = None

version = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if version > 0:
    learner.load_model_version(env, version)

past_agents = []

i = 0
while True:
    if epochs is not None and i >= epochs:
        print("Training Complete")
        break

    i = i + 1
    version += timesteps
    version = int(version)

    agent_data = {"version": version}

    print(f"Model {learner.model_name} Version {version} (i {i}) training vs Self")
    learner.learn(timesteps)

    learner.save(version)
    agent = learner.get_agent()
    agent_data["agent"] = agent
    print("Vs Random:")
    results_random = get_win_percentages(agent, "random")
    print_win_percentages(results_random)
    agent_data["vs_random"] = results_random.agent1_wins
    print("Vs Lookahead:")
    results_look = get_win_percentages(agent, multistep_agent_factory())
    agent_data["vs_look"] = results_look.agent1_wins
    print_win_percentages(results_look)

    print("Vs Negamax:")
    results_nega = get_win_percentages(agent, "negamax")
    agent_data["vs_nega"] = results_nega.agent1_wins

    print_win_percentages(results_nega)

    update_model_data(learner.model_name, version, results_random, results_look, results_negamax=results_nega)
    past_agents.append(agent_data)

    past_agents.sort(key=lambda x: x["vs_random"] * 0.1 + x["vs_look"] * 0.5 + x["vs_nega"], reverse=True)

    past_agents = past_agents[:5]

    print("Past Agents:")
    print(past_agents)

    if results_look.agent1_wins < 0.8:
        adv_agent = {"agent": multistep_agent_factory()}
    elif results_nega.agent1_wins < 0.2:
        adv_agent = {"agent": "negamax"}
    else:
        adv_agent = random.choice(past_agents)

    print("Next Agent: ")
    print(adv_agent)

    env = get_env(adv_agent["agent"])
    learner.load_model_version(env, version)
