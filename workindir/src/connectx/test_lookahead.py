import sys
from torch import nn as nn

from connectx.utils import get_win_percentages
from connectx.lookahead import multistep_agent_factory
from connectx.look_agent import agent_factory
from connectx.environment import (
    ConnectFourGymV2,
    ConnectFourGymV3,
    ConnectFourGymV4,
    ConnectFourGymV5,
    ConnectFourGymV6,
)


import random
import importlib


def get_env(agent2):
    env = ConnectFourGymV2(agent2=agent2)

    return env


env = get_env("random")


# agent = agent_factory()
agent = multistep_agent_factory(move_predict=3)

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


# print("VS negamax")
# win_percentage_1 = get_win_percentages(agent, "negamax")
# print(f"Vs Negamax (first): {win_percentage_1.agent1_wins}")
# win_percentage_2 = get_win_percentages("negamax", agent)
# print(f"Vs Negamax (second): {win_percentage_2.agent2_wins}")
