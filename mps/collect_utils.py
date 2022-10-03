import pdb
import json
import sys
import subprocess
import time
import os
import grpc
home = os.environ.get('HOME')
os.chdir(f'{home}/GIT/socc22-miso/mps/models')
sys.path.append(f'{home}/GIT/socc22-miso/mps/grpc')
import grpc_pb2, grpc_pb2_grpc
from concurrent import futures
import pandas as pd

port = 50053
data_dir = '/work/li.baol/MISO_utils'

with open(f'{home}/GIT/socc22-miso/mps/configs/batch.json') as f:
    batch_dict = json.load(f)

class Scheduler(grpc_pb2_grpc.SchedulerServicer):
    def __init__(self):
        super().__init__()
        self.pid = 0
        self.started = False
    def Notify(self, request, context):
        # update the list of ready jobs to send CONT signal
        cmd = f'kill -18 {self.pid}'
        subprocess.Popen([cmd], shell=True)
        print(f'Sent signal to PID {self.pid}')
        self.started = True
        return grpc_pb2.ServerReply(message='Start Measure Signal Sent')
    def update_pid(self, pid):
        self.pid = pid
    def reset_started(self):
        self.started = False

grpc_ins = Scheduler()

########## server ##########

server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
grpc_pb2_grpc.add_SchedulerServicer_to_server(grpc_ins, server)
server.add_insecure_port(f'[::]:{port}')
server.start()
#server.wait_for_termination()

###########################


memory_dict = {}
util_sm = {}
util_mem = {}
power = {}

for model, val in batch_dict.items():
    memory_dict[model] = {}
    util_sm[model] = {}
    util_mem[model] = {}
    power[model] = {}
    for batch in val:
        cmd = f'CUDA_VISIBLE_DEVICE=0 python {model}_train.py --gpu_type test -b {batch} --port {port}'
        print(cmd)
        proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        pid = proc.pid
        grpc_ins.update_pid(pid)
        # wait for it to start
        while not grpc_ins.started:
            time.sleep(0.1)
        time.sleep(10)
        grpc_ins.reset_started()
        # measure gpu memory usage using nvidia-smi
        cmd = f'timeout 5 nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,nounits -lms 100 \
            --filename={data_dir}/{model}_{batch}.csv'
        p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        mem, err = p.communicate()
        # read csv file
        df = pd.read_csv(f'{data_dir}/{model}_{batch}.csv')

        memory_dict[model][batch] = round(df['memory.used [MiB]'].mean(),2)
        util_sm[model][batch] = round(df[' utilization.gpu [%]'].mean(),2)
        util_mem[model][batch] = round(df[' utilization.memory [%]'].mean(),2)
        power[model][batch] = round(df[' power.draw [W]'].mean(), 2)

        cmd = f'kill -9 {pid}'
        subprocess.Popen([cmd], shell=True)
        proc.wait() # wait for it to finish
        print('process finished')

    with open(f'{home}/GIT/socc22-miso/mps/configs/memory_{model}.json', 'w') as f:
        json.dump(memory_dict[model], f, indent=4)
    with open(f'{home}/GIT/socc22-miso/mps/configs/util_sm_{model}.json', 'w') as f:
        json.dump(util_sm[model], f, indent=4)    
    with open(f'{home}/GIT/socc22-miso/mps/configs/util_mem_{model}.json', 'w') as f:
        json.dump(util_mem[model], f, indent=4)    
    with open(f'{home}/GIT/socc22-miso/mps/configs/power_{model}.json', 'w') as f:
        json.dump(power[model], f, indent=4)