#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: run-train.sh <model> <random | lookahead | negamax | model_name> [<version>]"
    exit 1
fi

model=$1
player2=$2

if [ "$#" -eq 3 ]; then
    version=$3
else
    version=""
fi

# model_control_120big_v13

docker run --gpus=all --runtime nvidia -v ./workindir:/workindir --rm kaggle-gpu /bin/bash train.sh $model $player2 $version 

echo "Stopping all containers"

docker stop $(docker ps | grep kaggle | awk '{print $1}')