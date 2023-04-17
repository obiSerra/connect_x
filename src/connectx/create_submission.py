import torch as th
import importlib
from os import path
import sys
from jinja2 import Environment, FileSystemLoader

from connectx.environment import ConnectFourGymV2


module = importlib.import_module(f"connectx.models.{sys.argv[1]}")
load_version = None
version = 0
if len(sys.argv) < 3:
    print("Usage: python create_submission.py <model_name> <version>")
    exit(1)

load_version = int(sys.argv[2])
version = load_version

outfile = sys.argv[3] if len(sys.argv) > 3 else "submission"
outfile = f"{outfile}.py"

env = ConnectFourGymV2()
model = module.get_model(env)


model.load_model_version(env, version)

print(model.model_name)

th.set_printoptions(profile="full")
state_dict_origin = model.learner.policy.to("cpu").state_dict()
policy_keys = state_dict_origin.keys()

state_dict = {}
for key in policy_keys:
    # print(key)
    if "policy_net" in key or "shared_net" in key:
        els = key.split(".")
        i = int(els[2])
        prefix = "policy" if "policy_net" in key else "shared"
        policy_index = i // 2 + 1
        state_dict[f"{prefix}{policy_index}.{els[-1]}"] = state_dict_origin[key]
    elif key.startswith("action"):
        state_dict[key.replace("_net", "")] = state_dict_origin[key]


p = path.join(path.dirname(__file__))

env = Environment(loader=FileSystemLoader(p))
template = env.get_template("submission_template.jinja.py")


output_submission = template.render(state_dict=state_dict)

with open(outfile, "w") as fh:
    fh.write(output_submission)

print(f"submission written to {outfile}")
