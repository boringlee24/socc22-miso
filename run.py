import pdb
import time
import os
import random
import json
import numpy as np
import glob
import argparse
import math
from pathlib import Path
import copy
from exp_full import Experiment
from exp_static import Static
from exp_oracle import Oracle
from exp_mps import MPS
from exp_miso import MISO
import time

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument('--arrival', type=int, help='inter-arrival period', default=60)
parser.add_argument('--num_job', type=int, help='total number of jobs', default=100)
parser.add_argument('--num_gpu', type=int, help='total number of GPUs', default=8)
parser.add_argument('--overhead', type=int, help='average migration overhead', default=30)
parser.add_argument('--seed', type=int, help='random seed', default=1) 
parser.add_argument('--error_mean', type=float, help='mean error of predictor', default=0.016)
parser.add_argument('--error_std', type=float, help='error variance when using Gaussian to generate prediction error', default=0.0016*2)
parser.add_argument('--test', action='store_true', help='test mode', default=False)
parser.add_argument('--step', type=int, help='simulation step size', default=10)
parser.add_argument('--filler', action='store_true', help='make first 5% of jobs as filler jobs', default=False)
parser.add_argument('--flat_arrival', action='store_true', help='do not make first 50 jobs arrive more frequently', default=False)
parser.add_argument('--random_trace', action='store_true', help='randomly sample jobs (specified by num_job) from the whole trace', default=False)
args = parser.parse_args()

Path('logs/miso').mkdir(parents=True, exist_ok=True)
Path('logs/full').mkdir(parents=True, exist_ok=True)
Path('logs/static').mkdir(parents=True, exist_ok=True)
Path('logs/oracle').mkdir(parents=True, exist_ok=True)
Path('logs/mps').mkdir(parents=True, exist_ok=True)

physical_nodes = ['d3104', 'd3105']#['d3103', 'd3100', 'd3101', 'd3106']

print('MISO')
miso_exp = MISO(args, physical_nodes)
miso_exp.run(args)
time.sleep(300)

print('Full')
full_exp = Experiment(args, physical_nodes)
full_exp.run(args)
time.sleep(300) # rest 5 min

print('Static')
static_exp = Static(args, physical_nodes)
static_exp.run(args)
time.sleep(300) # rest 5 min

print('Oracle')
oracle_exp = Oracle(args, physical_nodes)
oracle_exp.run(args)
time.sleep(300) # rest 5 min

print('MPS')
mps_exp = MPS(args, physical_nodes)
mps_exp.run(args)
