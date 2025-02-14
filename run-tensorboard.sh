#!/bin/bash

docker run \
    -d \
    -v ./workindir/logs/:/app/runs/ \
    -p 6006:6006 \
    --restart always \
    -w "/app/" \
    --name "tensorboard" \
    schafo/tensorboard