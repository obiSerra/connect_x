import sys
from torch import nn as nn

from connectx.utils import get_win_percentages
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


if len(sys.argv) < 3:
    print("usage $ python src/connectx/quick_test_model.py <model_name> <version>")
    exit(1)


version = sys.argv[2]

module = importlib.import_module(f"connectx.models.{sys.argv[1]}")


env = get_env("random")

model = module.get_model(env)

model.load_model_version(env, int(version))


agent = model.get_agent()

print("------------------   AGENT V2   ------------------")

print("VS random")
win_percentage_1 = get_win_percentages(agent, "random")
print(f"Vs Random (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages("random", agent)
print(f"Vs Random (second): {win_percentage_2.agent2_wins}")

print("VS lookahead")
win_percentage_1 = get_win_percentages(agent, multistep_agent_factory())
print(f"Vs Lookahead (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages(multistep_agent_factory(), agent)
print(f"Vs Lookahead (second): {win_percentage_2.agent2_wins}")

print("VS negamax")
win_percentage_1 = get_win_percentages(agent, "negamax")
print(f"Vs Negamax (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages("negamax", agent)
print(f"Vs Negamax (second): {win_percentage_2.agent2_wins}")


print("------------------   AGENT V3   ------------------")

agent = model.get_agentV3()

print("VS random")
win_percentage_1 = get_win_percentages(agent, "random")
print(f"Vs Random (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages("random", agent)
print(f"Vs Random (second): {win_percentage_2.agent2_wins}")

print("VS lookahead")
win_percentage_1 = get_win_percentages(agent, multistep_agent_factory())
print(f"Vs Lookahead (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages(multistep_agent_factory(), agent)
print(f"Vs Lookahead (second): {win_percentage_2.agent2_wins}")

print("VS negamax")
win_percentage_1 = get_win_percentages(agent, "negamax")
print(f"Vs Negamax (first): {win_percentage_1.agent1_wins}")
win_percentage_2 = get_win_percentages("negamax", agent)
print(f"Vs Negamax (second): {win_percentage_2.agent2_wins}")
