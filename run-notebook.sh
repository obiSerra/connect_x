#!/bin/bash

docker run -p 8888:8888 --gpus=all --runtime nvidia -v ./workindir:/workindir --rm kaggle-gpu