import pdb
import json
import sys
import subprocess
import time
import os
import grpc
home = os.environ.get('HOME')
os.chdir(f'{home}/GIT/mig_exp/mps/models')
sys.path.append(f'{home}/GIT/mig_exp/mps/grpc')
import grpc_pb2, grpc_pb2_grpc
from concurrent import futures

port = 50053

with open(f'{home}/GIT/mig_exp/mps/configs/batch.json') as f:
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

for model, val in batch_dict.items():
    memory_dict[model] = {}
    for batch in val:
        cmd = f'python {model}_train.py --gpu_type test -b {batch} --port {port}'       
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
        cmd = 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
        p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        mem, err = p.communicate()
        memory_dict[model][batch] = int(mem)
        cmd = f'kill -9 {pid}'
        subprocess.Popen([cmd], shell=True)
        proc.wait() # wait for it to finish
        print('process finished')

with open(f'{home}/GIT/mig_exp/mps/configs/memory.json', 'w') as f:
    json.dump(memory_dict, f, indent=4)
