import sys
from torch import nn as nn
from connectx.model_agent_loader import get_agent

from connectx.utils import get_win_percentages, print_win_percentages, update_model_data
from stable_baselines3.common.env_util import make_vec_env
from connectx.lookahead import multistep_agent_factory
from connectx.trainers.train_utils import load_module
from connectx.environment import (
    ConnectFourGymV10,
    ConnectFourGymV12,
    ConnectFourGymV2,
    ConnectFourGymV3,
    ConnectFourGymV4,
    ConnectFourGymV5,
    ConnectFourGymV6,
    ConnectFourGymV7,
    ConnectFourGymV8,
    ConnectFourGymV9,
)


def train(learner, epochs=None, version=0, timesteps=10e4):
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


MODEL_ADV_LIST = [("model_control_120big_v8", 14500000)]


def load_adv(mode_adv):
    advs = []
    for m in mode_adv:
        advs.append(get_agent(m[0], m[1]))
    return advs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage $ python src/train.py <model_name>")
        # print("usage $ python src/train.py <model_name> <model_version>")
        exit(1)

    module = load_module(sys.argv[1])

    env_list = {"v12": ConnectFourGymV12}

    selected_env = None
    for e in env_list.items():
        if sys.argv[1].endswith(e[0]):
            selected_env = e[1]

    if selected_env is None:
        print("No env found for model")
        exit(1)

    available_advs = [multistep_agent_factory(), "random"]

    available_advs += load_adv(MODEL_ADV_LIST)

    env = selected_env(adv_agents=available_advs)
    # env = make_vec_env(lambda: env, n_envs=4)
    print("[+] Selected env", selected_env)

    learner = module.get_learner(env)

    timesteps = 10e4

    # Basic Training

    train(learner, epochs=None, version=0, timesteps=timesteps)
