#!/usr/bin/env bash

i=1;
j=$#;
while [ $i -le $j ] 
do
    echo "Train model - $i: $1";
    "$(python src/connectx/trainers/adv_train.py $1 300000)"
    i=$((i + 1));
    shift 1;
done