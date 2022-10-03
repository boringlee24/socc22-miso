import pdb
import json
import sys
import subprocess
import time
import random
import os
from pathlib import Path
import grpc
import numpy as np
home = os.environ.get('HOME')
user = os.environ.get('USER')
os.chdir(f'{home}/GIT/socc22-miso/mps/models')
sys.path.append(f'{home}/GIT/socc22-miso/mps/grpc')
import grpc_pb2, grpc_pb2_grpc
from concurrent import futures
import pandas as pd
import psutil
import argparse

parser = argparse.ArgumentParser(description='MPS')
parser.add_argument('-i', '--index', metavar='GPU_index', type=str) # default 0
parser.add_argument('--port', type=int) # default 50051
parser.add_argument('--start', metavar='STARTING_REPEAT', type=int, default=200) #TODO
parser.add_argument('--repeat', metavar='REPEAT', type=int, default=400)
parser.add_argument('--num_active', nargs='+', help='<Required> number of active slices', required=True, type=int)

args = parser.parse_args()

os.environ['CUDA_MPS_PIPE_DIRECTORY'] = f'/scratch/{user}/mps_log/nvidia-mps{args.index}'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = f'/scratch/{user}/mps_log/nvidia-log{args.index}'

num_active = args.num_active

fractions = [100, 50, 14]
with open(f'{home}/GIT/socc22-miso/mps/configs/batch.json') as f:
    batch_dict = json.load(f)
with open(f'{home}/GIT/socc22-miso/mps/configs/memory.json') as f:
    mem_dict = json.load(f)

repeat = args.repeat 
port = args.port
time_lim = 60 #TODO
start_rep = args.start 

class Scheduler(grpc_pb2_grpc.SchedulerServicer):
    def __init__(self):
        super().__init__()
        self.record = []
    def Notify(self, request, context):
        # update the list of ready jobs to send CONT signal
        self.record.append(f'{request.model}_{request.batch}_{request.mode}')
#        cmd = f'kill -18 {self.pid}'
#        subprocess.Popen([cmd], shell=True)
        print(f'Sent signal to {request.model}_{request.batch}_{request.mode}')
        return grpc_pb2.ServerReply(message='Start Measure Signal Sent')
    def clear_record(self):
        self.record = []
grpc_ins = Scheduler()

########## server ##########

server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
grpc_pb2_grpc.add_SchedulerServicer_to_server(grpc_ins, server)
server.add_insecure_port(f'[::]:{port}') 
server.start()
#server.wait_for_termination() #TODO

###########################

for num in num_active:
    for k in range(start_rep, repeat):
        Path(f'/scratch/{user}/mps_collect/active{num}_repeat{k}').mkdir(parents=True,exist_ok=True)
        cmd = f'rm -f /scratch/{user}/mps_collect/active{num}_repeat{k}/*'
        del_proc = subprocess.Popen([cmd], shell=True)
        del_proc.wait()

        mem_limit = 38912 # 38GB
        while True:
            active_jobs = []
            label = ''
            used_mem = 0
            for i in range(num):
                key = random.choice(list(batch_dict)) 
                val = random.choice(batch_dict[key])
                active_jobs.append((key, val))
                label += key[0]
                used_mem += mem_dict[key][str(val)]
            while len(active_jobs) < 7:
                active_jobs.append('dummy')
                label += 'y'
                used_mem += mem_dict['dummy']['32'] 
            if used_mem <= mem_limit:
                break
        print(f'Starting repeat {k} {label}, memory is {used_mem}MB')
        for frac in fractions:
            pid_list = []
            record_list = []
            proc_list = []
            for ind, job in enumerate(active_jobs):
                mode = f'{label}{ind}_{frac}pct'
                if job == 'dummy':
                    cmd = f'python {job}_train.py --gpu_type {mode} --partition {frac} --time_limit {time_lim} --port {port}'
                    record_list.append(f'dummy_train_32_{mode}')
                else:
                    model, batch = job
                    cmd = f'python {model}_train.py --gpu_type {mode} --partition {frac} -b {batch} --time_limit {time_lim} --port {port}'
                    record_list.append(f'{model}_train_{batch}_{mode}')
                print(cmd)
                proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                pid = proc.pid
                pid_list.append(pid)
                proc_list.append(proc)
            # wait till all started jobs appear in the grpc record
            while set(record_list) != set(grpc_ins.record):
                time.sleep(2)
            grpc_ins.clear_record()
            # start all pids
            for process in pid_list:
                cmd = f'kill -18 {process}'
                subprocess.Popen([cmd], shell=True)
            # wait for processes to finish
            for proc in proc_list:
                proc.wait()
            time.sleep(2)
            print(f'repeat {k} done on {label} {frac}%')
        # post-process collected data as 3x7 matrix, then dump data to scratch
        arr = np.zeros((3,7))
        for col, job in enumerate(active_jobs):
            for row, frac in enumerate(fractions):
                if job == 'dummy':
                    filename = f'logs/{label}{col}_{frac}pct_{job}_train32.json'
                else:
                    filename = f'logs/{label}{col}_{frac}pct_{job[0]}_train{job[1]}.json'
                with open(filename) as f:
                    lat = json.load(f)
                mean_lat = []
                for key, val in lat.items():
                    mean_lat += val
                mean_lat = mean_lat[1:] # remove 1st element
                mean_lat = round(np.mean(mean_lat),4)
                arr[row,col] = mean_lat
                cmd = f'mv {filename} /scratch/{user}/mps_collect/active{num}_repeat{k}/'
                subprocess.Popen([cmd], shell=True)
        df = pd.DataFrame(arr)

        cmd = f'rm -f logs/active{num}_repeat{k}_*.csv'
        del_proc = subprocess.Popen([cmd], shell=True)
        del_proc.wait()

        df.to_csv(f'logs/active{num}_repeat{k}_{label}.csv', header=False, index=False)
#        pdb.set_trace()
#        np.savetxt(f'logs/active{num}_repeat{k}_{label}.csv', arr, delimiter=",")
        print(f'save to file logs/active{num}_repeat{k}_{label}.csv')

