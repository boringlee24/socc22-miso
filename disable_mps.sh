#!/bin/bash

echo quit | nvidia-cuda-mps-control
pkill -9 nvidia-cuda-mps

sudo nvidia-smi -i 0,1 -c DEFAULT
