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

mode = sys.argv[1] # 7g.40gb

class Scheduler(grpc_pb2_grpc.SchedulerServicer):
    def __init__(self):
        super().__init__()
        self.record = []
        self.pid = 0
    def Notify(self, request, context):
        # update the list of ready jobs to send CONT signal
        self.record.append(f'{request.model}_{request.batch}_{request.mode}')
        cmd = f'kill -18 {self.pid}'
        subprocess.Popen([cmd], shell=True)
        print(f'Sent signal to PID {self.pid}')
        return grpc_pb2.ServerReply(message='Start Measure Signal Sent')
    def update_pid(self, pid):
        self.pid = pid
    def remove_record(self, item):
        self.record.remove(item)
grpc_ins = Scheduler()

########## server ##########

server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
grpc_pb2_grpc.add_SchedulerServicer_to_server(grpc_ins, server)
server.add_insecure_port('[::]:50051')
server.start()
#server.wait_for_termination()

###########################

#TODO
if mode == '1g.5gb':
    gpuid = 'MIG-fe34059c-48a8-5b06-b049-57d4202844b5'
elif mode == '2g.10gb':
    gpuid = 'MIG-c844d51d-8b9b-5b8e-a04b-296ebdf3b43b'
elif mode == '3g.20gb':
    gpuid = 'MIG-e986fa58-fbb1-54bd-8bf9-20d6d2ef0ddd'
elif mode == '7g.40gb':
    gpuid = 'GPU-b66ca3f2-4f2b-1dfb-5aa1-4418b3bccad0'
elif mode == '4g.20gb':
    gpuid = 'MIG-2dc3bfd1-6716-5ba8-a9b1-d4922d6aaa79'

os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
g = mode.split('.')[0]
gb = mode.split('.')[1]

with open(f'{home}/GIT/socc22-miso/mps/configs/{g}_{gb}.json') as f:
    batch_dict = json.load(f)

for model,batches in batch_dict.items():
    if model not in ['resnet', 'mobilenet']: #TODO
        for batch in batches:
            cmd = f'python {model}_train.py --gpu_type {mode} -b {batch}'
            print(cmd)
            proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            pid = proc.pid
            grpc_ins.update_pid(pid)
            proc.wait() # wait for it to finish
            if f'{model}_train_{batch}_{mode}' not in grpc_ins.record:
                pdb.set_trace()
            grpc_ins.remove_record(f'{model}_train_{batch}_{mode}')
            print(f'{model} {batch} done on {mode}')
            time.sleep(5)
