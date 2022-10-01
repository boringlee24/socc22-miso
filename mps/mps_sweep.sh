#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=/scratch/$USER/mps_log/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/scratch/$USER/mps_log/nvidia-log
#export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1

cd models

MPS_MODE=test_100pct
#export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
python resnet_train.py --gpu_type $MPS_MODE --direct_start --partition 100 --time_limit 30 -b 512 &&

MPS_MODE=test_50pct
#export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
python resnet_train.py --gpu_type $MPS_MODE --direct_start --partition 50 --time_limit 30 -b 512 &&

MPS_MODE=test_14pct
#export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14
python resnet_train.py --gpu_type $MPS_MODE --direct_start --partition 14 --time_limit 30 -b 512

#python resnet_train.py --gpu_type $MPS_MODE --partition 40
#python deepspeech2_train.py --gpu_type $MPS_MODE 

#
#MODEL_LIST=("resnet"  "resnet"  "deepspeech2"  "transformer"  "gnn"  "embedding")
#
#cd ../models
#mkdir -p logs/mps
#
#for val in ${MODEL_LIST[*]}; do
#    echo $val;
#    python ${val}_train.py --gpu_type "${MPS_MODE}" &&
#    sleep 30;
#    python ${val}_inf.py --gpu_type ${MIG_MODE} &&
#    sleep 30;
#    echo done with $val
#done   
#
#echo finished
#
