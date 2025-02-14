import os

import torch as th

from connectx.BaseModel import BaseModelV2, BaseModelV3


model_name = os.path.basename(__file__).replace(".py", "")


policy_kwargs = {
    "activation_fn": th.nn.ReLU,
    "net_arch": [dict(pi=[240, 240, 240, 240], vf=[240, 240, 240, 240])],
}

model_params = {
    "batch_size": 240,
    "n_steps": 480,
    "policy_kwargs": policy_kwargs,
    "seed": 42,
}


def get_learner(env, new_model=True):
    return BaseModelV2(
        env, model_name, model_params, policy="MultiInputPolicy", new_model=new_model
    )


def get_model(env):
    return BaseModelV2(
        env,
        model_name,
        model_params,
        policy="MultiInputPolicy",
        new_model=False,
        print_model=False,
    )


def get_modelV3(env):
    return BaseModelV3(
        env,
        model_name,
        model_params,
        policy="MultiInputPolicy",
        new_model=False,
        print_model=False,
    )
