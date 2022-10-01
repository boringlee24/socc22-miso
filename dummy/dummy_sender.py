import pdb
import time
import os
user = os.environ.get('USER')
import re
import random
import json
import numpy as np
import glob
import socket
import argparse
import signal
import sys
sys.path.append(f'/home/{user}/GIT/mig_exp/workloads/')
from send_signal import send_signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--node', metavar='GPU_NODE', type=str, help='specific which node', default='d3106')
    parser.add_argument('--port', metavar='PORT_NUMBER', type=int, help='select which port for communication', default=10000)
    parser.add_argument('--cmd', type=str, help='command to send')
    args = parser.parse_args()

    send_signal(args.node, args.port, args.cmd)
