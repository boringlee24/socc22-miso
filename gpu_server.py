import socket
import sys
import pdb
import subprocess
import re
import json
import os
user = os.environ.get('USER')
sys.path.append(f'/home/{user}/GIT/socc22-miso/mps/scheduler/simulator/')
from utils import *
import argparse
import psutil
import time
import datetime
import mig_helper
import signal
from pathlib import Path
os.chdir('workloads')

parser = argparse.ArgumentParser(description='TCP server')
parser.add_argument('--node', metavar='GPU_NODE', type=str, help='specific which node', default=socket.gethostname())
parser.add_argument('--port', metavar='PORT_NUMBER', type=int, default=10000, help='select which port for communication')
parser.add_argument('--host', metavar='HOST_NODE', type=str, help='scheduler node', default='invalid')
#parser.add_argument('--tc', type=str, help='testcase', default='test') # miso, full

args = parser.parse_args()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = (args.node, args.port)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(5)


with open(f'/home/{user}/GIT/socc22-miso/mig_device_autogen.json') as f:
    cuda_devices = json.load(f)[args.node]
with open(f'/home/{user}/GIT/socc22-miso/mps/scheduler/simulator/job_models.json') as f:
    job_models = json.load(f)
with open(f'/home/{user}/GIT/socc22-miso/workloads/num_iters.json') as f:
    num_iters = json.load(f)
with open(f'/home/{user}/GIT/socc22-miso/mps/scheduler/partition_code.json') as f:
    partition_code = json.load(f)

run_pid_dict = {}

current_partition = {0: '0', 1: '0'}
host_node = ''
log_dir = f'/scratch/{user}/miso_logs'

def catchInterrupt(signalNumber, frame):
    print('received SIGINT, no action needed')

signal.signal(signal.SIGINT, catchInterrupt)

def pid_finished(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    else:
        return False

while True:
    # Wait for a connection
    connection, client_address = sock.accept()
    try:
        # keep receiving until nothing is received 
        while True:
            data = connection.recv(40)
            if data:
                data_str = data.decode('utf-8')
                print('received ' + data_str)
                if 'start' in data_str or 'resume' in data_str: # 'start 15 gpu 1 slice 0'
                    jobid = re.findall(r'\d+', data_str)[0]
                    gpuid = int(re.findall(r'\d+', data_str)[1])
                    sliceid = int(re.findall(r'\d+', data_str)[2])
                    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][sliceid]
                    mapped_jobid = str(int(jobid) % 100)
                    model = job_models[mapped_jobid].split('_')[0]
                    batch = job_models[mapped_jobid].split('train')[1]
                    iters = num_iters[mapped_jobid]
                    if 'start' in data_str:
                        cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} --node {host_node}'
                        print(f'starting job {jobid} at gpu {gpuid} slice {sliceid}')
                        out_file = f'{log_dir}/job{jobid}_start.out'
                        err_file = f'{log_dir}/job{jobid}_start.err'
                    else:
                        resume_batch = int(re.findall(r'\d+', data_str)[3])
                        cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} --node {host_node} --resume --start_batch {resume_batch}'
                        print(f'resuming job {jobid} at gpu {gpuid} slice {sliceid} batch {resume_batch}')

                        out_file = f'{log_dir}/job{jobid}_resume.out'
                        err_file = f'{log_dir}/job{jobid}_resume.err'
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
                elif 'mps_strt' in data_str: # mps_strt 10 gpu 0 lvl 100
                    jobid = re.findall(r'\d+', data_str)[0]
                    gpuid = int(re.findall(r'\d+', data_str)[1])
                    mps_lvl = re.findall(r'\d+', data_str)[2]
                    if current_partition[gpuid] != '0':
                        raise RuntimeError('GPU must be in 7g.40gb to start MPS')
                    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
                    mapped_jobid = str(int(jobid) % 100)
                    model = job_models[mapped_jobid].split('_')[0]
                    batch = job_models[mapped_jobid].split('train')[1]
                    iters = num_iters[mapped_jobid]
                    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
                        --node {host_node} --partition {mps_lvl} --mps_set --cuda_device {device}'
                    print(f'starting job {jobid} at gpu {gpuid} for MPS')
                    
                    out_file = f'{log_dir}/job{jobid}_start.out'
                    err_file = f'{log_dir}/job{jobid}_start.err'
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)
                elif 'mps_rsm' in data_str: # mps_rsm 10 gpu 0 batch 500 lvl 100
                    jobid = re.findall(r'\d+', data_str)[0]
                    gpuid = int(re.findall(r'\d+', data_str)[1])
                    resume_batch = int(re.findall(r'\d+', data_str)[2])
                    mps_lvl = re.findall(r'\d+', data_str)[3]
                    if current_partition[gpuid] != '0':
                        raise RuntimeError('GPU must be in 7g.40gb to start MPS')
                    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
                    mapped_jobid = str(int(jobid) % 100)
                    model = job_models[mapped_jobid].split('_')[0]
                    batch = job_models[mapped_jobid].split('train')[1]
                    iters = num_iters[mapped_jobid]
                    cmd = f'CUDA_VISIBLE_DEVICES={device} python {model}_train.py --job_id {jobid} -b {batch} --iters {iters} \
                        --node {host_node} --partition {mps_lvl} --mps_set --resume --start_batch {resume_batch} --cuda_device {device}'
                    print(f'resuming job {jobid} at gpu {gpuid} for MPS')

                    out_file = f'{log_dir}/job{jobid}_resume.out'
                    err_file = f'{log_dir}/job{jobid}_resume.err'     
                    with open(out_file, 'w+') as out, open(err_file, 'w+') as err:
                        subprocess.Popen([cmd], shell=True, stdout=out, stderr=err)                                   
                elif 'mps_enable' in data_str: # mps_enable 0
                    gpuid = int(re.findall(r'\d+', data_str)[0])
                    mig_helper.reset_mig(gpuid)
                    mig_helper.create_ins(gpuid, '7g.40gb')
                    current_partition[gpuid] = '0'
                    device = cuda_devices[f'gpu{gpuid}'][current_partition[gpuid]][0]
                    cmd = f'./enable_mps_on_mig.sh {device}'
                    p = subprocess.Popen([cmd], shell=True)
                    p.wait()
                    print(f'enabled MPS on GPU {gpuid}')
                elif 'mps_disable' in data_str: # mps_disable 0
                    gpuid = int(re.findall(r'\d+', data_str)[0])
                    if current_partition[gpuid] != '0':
                        raise RuntimeError('When disabling MPS, the MIG partition is not 7g.40gb')
                    cmd = f'nvidia-smi -i {gpuid} --query-compute-apps=pid,process_name --format=csv,noheader'
                    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out_p, err_p = p.communicate()
                    out_str = out_p.decode('utf-8').strip().split('\n')
                    mps_pid = ''
                    if len(out_str) > 1:
                        # find the pid for mps
                        for row in out_str:
                            if 'nvidia-cuda-mps-server' in row:
                                mps_pid = row.split(', ')[0]
                                break
                        # raise RuntimeError('When disabling MPS, no job should be running')
                    # elif len(out_str) == 0:
                    #     raise RuntimeError('Cannot parse PID of MPS')
                    else:
                        mps_pid = out_str[0].split(', ')[0]
                    if mps_pid == '':
                        print(f'Potential Error: for some reason MPS is already disabled on GPU {gpuid}')
                    mps_ppid = psutil.Process(int(mps_pid)).ppid()
                    cmd = f'kill -9 {mps_ppid}'
                    p = subprocess.Popen([cmd], shell=True)
                    p.wait()
                    print(f'disabled MPS on GPU {gpuid}')
                elif 'config' in data_str: # 'config gpu 0 partition 6
                    gpuid = int(re.findall(r'\d+', data_str)[0])
                    code = re.findall(r'\d+', data_str)[1]
                    mig_helper.reset_mig(gpuid)
                    print(f'gpu {gpuid} has be reset')
                    partition = partition_code[code]
                    current_partition[gpuid] = code
                    for p in partition:
                        sliceid = GPU_status.num_to_str[p]
                        mig_helper.create_ins(gpuid, sliceid)
                    print(f'configured gpu {gpuid} to partition code {code}')
                elif 'kill all' == data_str: # 'kill all'
                    cmd = 'pkill -2 python'
                    subprocess.Popen([cmd], shell=True)
                    cmd = '../disable_mps.sh'
                    subprocess.Popen([cmd], shell=True)                    
                elif 'fkill' in data_str: # fkill 15 pid 19999
                    jobid = data_str.split(' ')[1]
                    pid = data_str.split(' ')[3]
                    if not pid_finished(int(pid)):
                        cmd = 'kill -2 ' + pid
                        subprocess.Popen([cmd], shell=True)
                        print('forced quit job' + jobid)
                elif 'save' in data_str: # 'save 15 pid 10000'
                    jobid = data_str.split(' ')[1]
                    pid = data_str.split(' ')[3]
                    cmd = 'kill -15 ' + pid
                    subprocess.Popen([cmd], shell=True)
                    print('checkpointing job' + jobid)
                elif 'hostname' in data_str: # 'hostname c2103'
                    host_node = data_str.split(' ')[1]
                elif 'log_dir' in data_str: # 'logdir static'
                    tc = data_str.split(' ')[1]
                    log_dir = f'/scratch/{user}/miso_logs/{tc}' # this is dir for training progress
                    Path(log_dir).mkdir(parents=True, exist_ok=True)

#                elif 'kill' in data_str: # 'kill 15', kills the run.sh processes
#                    jobid = re.findall(r'\d+', data_str)[0]
#                    run_pid = run_pid_dict['job'+jobid]
#                    cmd = 'pkill -15 -P ' + str(run_pid)
#                    print('sending kill command to run.sh PID ' + str(run_pid))
#                    subprocess.Popen([cmd], shell=True)

                connection.sendall(b'success')
            else:
                break

    finally:
        # Clean up the connection
        connection.close()
