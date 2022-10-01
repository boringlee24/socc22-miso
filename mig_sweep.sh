#!/bin/bash

# TODO: 1. reconfigure mig instances 2. set CUDA_VISIBLE...

MIG_MODE="v100"
MODEL_LIST=("resnet"  "bert"  "deepspeech2"  "transformer"  "gnn"  "embedding")

cd models

for val in ${MODEL_LIST[*]}; do
    echo $val;
    python ${val}_train.py --gpu_type ${MIG_MODE} &&
    sleep 30;
    python ${val}_inf.py --gpu_type ${MIG_MODE} &&
    sleep 30;
    echo done with $val
done   

echo finished

