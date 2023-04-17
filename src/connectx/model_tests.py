import random
from copy import deepcopy
import os
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
)


import importlib


# module = importlib.import_module(f'connectx.models.{sys.argv[1]}')

dir_path = "data"

# list to store files
res = []


def get_best_model(file_name, models_to_test=None):
    points = 0
    best = None
    with open(file_name) as csvfile:
        for line in csvfile:
            info = line.split(",")
            try:
                vs_random = float(info[2])
                vs_adv = float(info[4])
                # model_points = (vs_random + 0.1) + vs_adv
                model_points = vs_adv
                if model_points > points and (models_to_test is None or info[0] in models_to_test):
                    best = {"name": info[0], "version": info[1], "points": model_points}
                    points = model_points
            except ValueError:
                pass
    print(f"Best model: {best}")
    return best


def get_models(models_to_test=None):
    models = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        file_name = os.path.join(dir_path, path)
        if os.path.isfile(file_name) and path.endswith(".csv"):

            models.append(get_best_model(file_name, models_to_test=models_to_test))
    return [model for model in models if model is not None]


def load_agents(models):
    agents_orig = []

    for to_load_model in models:
        model_name = to_load_model["name"]
        if model_name.endswith("v3"):
            print("Using V3")
            env_basic = ConnectFourGymV3()
        elif model_name.endswith("v4"):
            print("Using V4")
            env_basic = ConnectFourGymV4()
        elif model_name.endswith("v5"):
            print("Using V5")
            env_basic = ConnectFourGymV5()
        elif model_name.endswith("v6"):
            print("Using V6")
            env_basic = ConnectFourGymV6()
        else:
            print("Using V2")
            env_basic = ConnectFourGymV2()

        print("[+] Loading model: ", to_load_model["name"], to_load_model["version"])
        module = importlib.import_module(f'connectx.models.{to_load_model["name"]}')
        model = module.get_model(env_basic)

        model.load_model_version(env_basic, int(to_load_model["version"]))
        agent = model.get_agent()
        agents_orig.append(
            {
                "model": to_load_model["name"],
                "agent": agent,
                "version": to_load_model["version"],
            }
        )
    return agents_orig


def agent_championship(models_to_test=None):
    models = get_models(models_to_test=models_to_test)

    wins = {}
    agents_orig = load_agents(models)

    print("Starting championship")
    for agent in agents_orig:
        wins[agent["model"]] = wins.get(agent["model"], {"w": 0, "l": [], "d": [], "version": agent["version"]})
        for adv_agent in agents_orig:
            if adv_agent["model"] == agent["model"]:
                continue
            print("Playing: ", agent["model"], " vs ", adv_agent["model"])
            result = get_win_percentages(agent["agent"], adv_agent["agent"])
            if result.agent1_wins > result.agent2_wins:
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": [], "version": agent["version"]})
                a1["w"] += 1
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": [], "version": adv_agent["version"]})
                a2["l"].append(agent["model"])
                wins[adv_agent["model"]] = a2
                print(agent["model"], " wins")
            elif result.agent1_wins < result.agent2_wins:
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": [], "version": agent["version"]})
                a1["l"].append(adv_agent["model"])
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": [], "version": adv_agent["version"]})
                a2["w"] += 1
                wins[adv_agent["model"]] = a2
                print(adv_agent["model"], " wins")
            else:
                print("Draw", result.agent1_wins, result.agent2_wins)
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": [], "version": agent["version"]})
                a1["d"].append(adv_agent["model"])
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": [], "version": adv_agent["version"]})
                a2["d"].append(agent["model"])
                wins[adv_agent["model"]] = a2

    return wins


def print_results(results):
    # list_results = [(k, v) for k, v in results.items()]

    list_results = []
    for k, v in results.items():
        v["points"] = v["w"] * 3 + len(v["d"])
        list_results.append((k, v))

    list_results.sort(key=lambda x: x[1]["points"], reverse=True)
    for (k, v) in list_results:
        print(f"{k}: {v}")


if __name__ == "__main__":
    # models_to_test = None

    results = agent_championship(models_to_test=None)

    print_results(results)
