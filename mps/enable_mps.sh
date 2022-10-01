#!/bin/bash

sudo nvidia-smi -i $1 -c EXCLUSIVE_PROCESS

export CUDA_VISIBLE_DEVICES=0

export CUDA_MPS_PIPE_DIRECTORY=/scratch/$USER/mps_log/nvidia-mps$1
export CUDA_MPS_LOG_DIRECTORY=/scratch/$USER/mps_log/nvidia-log$1

nvidia-cuda-mps-control -d
