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
    ConnectFourGymV10,
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


def load_module(module_name):
    module = importlib.import_module(f"connectx.models.{module_name}")
    return module


def get_agent(module_name, version):
    module = load_module(module_name)

    env = get_env("random")

    model = module.get_model(env)

    model.load_model_version(env, int(version))

    return model.get_agent()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage $ python src/connectx/quick_test_model.py <model_name> <version>")
        exit(1)

    agent = get_agent(sys.argv[1], sys.argv[2])
    print(agent)
