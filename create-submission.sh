#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: run-train.sh <model> <random | lookahead | negamax | model_name> [<version>]"
    exit 1
fi

model=$1
version=$2

# model_control_120big_v13

docker run --gpus=all --runtime nvidia -v ./workindir:/workindir --rm kaggle-gpu python src/connectx/create_submission.py $model $version