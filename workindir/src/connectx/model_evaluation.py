import random
from copy import deepcopy
import os
import sys
from torch import nn as nn

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from connectx.lookahead import multistep_agent_factory
from connectx.environment import ConnectFourGymV2, ConnectFourGymV3, ConnectFourGymV4, ConnectFourGymV5


import importlib
import numpy as np


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
                model_points = (vs_random + 0.5) + vs_adv
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

    env_basic = ConnectFourGymV2()
    for to_load_model in models:
        print("[+] Loading model: ", to_load_model["name"], to_load_model["version"])

        if to_load_model["name"].endswith("v3"):
            print("Using V3")
            env_basic = ConnectFourGymV3()
        elif to_load_model["name"].endswith("v4"):
            print("Using V4")
            env_basic = ConnectFourGymV4()
        elif to_load_model["name"].endswith("v5"):
            print("Using V5")
            env_basic = ConnectFourGymV5()
        else:
            print("Using V2")
            env_basic = ConnectFourGymV2()

        module = importlib.import_module(f'connectx.models.{to_load_model["name"]}')
        model = module.get_model(env_basic)

        model.load_model_version(env_basic, int(to_load_model["version"]))
        agent = model.get_agent()
        agents_orig.append(
            {
                "model": to_load_model["name"],
                "agent": agent,
            }
        )
    return agents_orig


def agent_championship(models_to_test=None):
    models = get_models(models_to_test=models_to_test)

    wins = {}
    agents_orig = load_agents(models)

    print("Starting championship")
    for agent in agents_orig:
        wins[agent["model"]] = wins.get(agent["model"], {"w": 0, "l": [], "d": []})
        for adv_agent in agents_orig:
            if adv_agent["model"] == agent["model"]:
                continue
            print("Playing: ", agent["model"], " vs ", adv_agent["model"])
            result = get_win_percentages(agent["agent"], adv_agent["agent"])
            if result.agent1_wins > result.agent2_wins:
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": []})
                a1["w"] += 1
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": []})
                a2["l"].append(agent["model"])
                wins[adv_agent["model"]] = a2
                print(agent["model"], " wins")
            elif result.agent1_wins < result.agent2_wins:
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": []})
                a1["l"].append(adv_agent["model"])
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": []})
                a2["w"] += 1
                wins[adv_agent["model"]] = a2
                print(adv_agent["model"], " wins")
            else:
                a1 = wins.get(agent["model"], {"w": 0, "l": [], "d": []})
                a1["d"].append(adv_agent["model"])
                wins[agent["model"]] = a1
                a2 = wins.get(adv_agent["model"], {"w": 0, "l": [], "d": []})
                a2["d"].append(agent["model"])
                wins[adv_agent["model"]] = a2

    return wins


def print_results(results):
    list_results = [(k, v) for k, v in results.items()]
    list_results.sort(key=lambda x: x[1]["w"], reverse=True)
    for (k, v) in list_results:
        print(f"{k}: {v}")


if __name__ == "__main__":

    models = get_models()
    model_agents = load_agents(models)
    print("")
    print(" --- ")
    print("")

    eval_repo = [["Model", "Random", "Lookahead", "Negamax"]]
    for model_agent in model_agents:
        model_row = [model_agent["model"]]
        wins = get_win_percentages(model_agent["agent"], "random")
        print(f"{model_agent['model']}: {wins.agent1_wins} vs random")
        model_row.append(wins.agent1_wins)

        wins = get_win_percentages(model_agent["agent"], multistep_agent_factory())
        print(f"{model_agent['model']}: {wins.agent1_wins} vs lookahead")
        model_row.append(wins.agent1_wins)

        wins = get_win_percentages(model_agent["agent"], "negamax")
        print(f"{model_agent['model']}: {wins.agent1_wins} vs negamax")
        model_row.append(wins.agent1_wins)
        eval_repo.append(model_row)

    np.savetxt("model_evaluation.csv", eval_repo, delimiter=", ", fmt="% s")
