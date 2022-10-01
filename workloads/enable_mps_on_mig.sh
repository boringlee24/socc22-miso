#!/bin/bash

sudo nvidia-smi -i $1 -c EXCLUSIVE_PROCESS
hs=`hostname`

export CUDA_VISIBLE_DEVICES=$1

mkdir -p /scratch/$USER/mps_log/nvidia-mps-$hs/$1
mkdir -p /scratch/$USER/mps_log/nvidia-log-$hs/$1

export CUDA_MPS_PIPE_DIRECTORY=/scratch/$USER/mps_log/nvidia-mps-$hs/$1
export CUDA_MPS_LOG_DIRECTORY=/scratch/$USER/mps_log/nvidia-log-$hs/$1

nvidia-cuda-mps-control -d
