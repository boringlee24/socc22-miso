import pdb
import time
import os
user = os.environ.get('USER')
import random
import json
import numpy as np
import glob
import argparse
import math
from pathlib import Path
import sys
sys.path.append(f'/home/{user}/GIT/mig_exp/mps/scheduler/simulator/')
from utils import *
import copy
sys.path.append(f'/home/{user}/GIT/mig_exp/workloads/')
from send_signal import send_signal
import socket
import threading
import _thread
import signal
from tcp_interpreter import *

# start job
def start_job(node, job, gpu, sliceid):
    cmd = f'start {job} gpu {gpu} slice {sliceid}'
    send_signal(node, cmd=cmd)   

def mps_start(node, job, gpu, level=50):
    cmd = f'mps_strt {job} gpu {gpu} lvl {level}'
    send_signal(node, cmd=cmd)   

def mps_resume(node, job, gpu, resume_batch, level=50):
    cmd = f'mps_rsm {job} gpu {gpu} batch {resume_batch} lvl {level}'
    send_signal(node, cmd=cmd)   

def resume_job(node, job, gpu, sliceid, resume_batch):
    cmd = f'resume {job} gpu {gpu} slice {sliceid} batch {resume_batch}'
    send_signal(node, cmd=cmd)   

def config_gpu(node, gpu, partition):
    cmd = f'config gpu {gpu} partition {partition}'
    send_signal(node, cmd=cmd)

def start_mps(node, gpu): # this is essentially reset_mig and create_ins 7g.40gb, then run ./enable_mps_on_mig
    cmd = f'mps_enable {gpu}'
    send_signal(node, cmd=cmd)

def end_mps(node, gpu): 
    cmd = f'mps_disable {gpu}'
    send_signal(node, cmd=cmd)

def fkill_job(node, job, pid):
    cmd = f'fkill {job} pid {pid}'
    send_signal(node, cmd=cmd)

def kill_all(node):
    cmd = 'kill all'
    send_signal(node, cmd=cmd)

def broadcast_host(node, runtime):
    cmd = f'hostname {socket.gethostname()}'
    send_signal(node, cmd=cmd)
    cmd = f'log_dir {runtime.tc}'
    send_signal(node, cmd=cmd)

def save_jobs(node, job_list, runtime, run_log):
    finish_status = [runtime.finish[job] for job in job_list]
    if 1 in finish_status:
        print(f'checkpoing {job_list} is invalid due to finish status {finish_status}', file=run_log, flush=True) #TODO: see if something can be done for jobs that are near finishing
        return False # indicate this save is invalid
    for job in job_list:
        if runtime.pid_dict[job] == 0:
            raise RuntimeError(f'job {job} PID is not updated')
        cmd = f'save {job} pid {runtime.pid_dict[job]}'
        send_signal(node, cmd=cmd)
    # wait till all ckpt signals are received
    print(f'waiting for checkpoint to finish, jobs {str(job_list)}', file=run_log, flush=True)
    while True:        
        # make sure all ckpt_dict are 1
        time.sleep(0.5) #TODO: 1 pdb.set_trace()
        ckpt_sum = 0
        for job in job_list:
            ckpt_sum += runtime.ckpt_dict[job]      
        if ckpt_sum == len(job_list):
            print('checkpoint finished', file=run_log, flush=True) 
            for job in job_list:
                fkill_job(node, job, runtime.pid_dict[job])
            return True

def thread_func(runtime, run_log, mode='full'): # this is an instance of the Experiment class 
    # here listen on the socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (socket.gethostname(), 10002)
    print('starting up on {} port {}'.format(*server_address), file=run_log, flush=True)
    sock.bind(server_address)
    sock.listen(5) 

    while True:
        # Wait for a connection
        connection, client_address = sock.accept()      
        try:
            while True:
                data = connection.recv(32)
                if data: 
                    data_str = data.decode('utf-8')
                    if 'term_thread' in data_str:
                        connection.sendall(b'success')
                        connection.close()
                    elif mode == 'full':
                        interpret_full(data_str, runtime, run_log)
                    elif mode == 'mps':
                        interpret_mps(data_str, runtime, run_log)                        
                    elif mode == 'static':
                        interpret_static(data_str, runtime, run_log)
                    elif mode == 'oracle' or mode == 'miso':
                        interpret_miso(data_str, runtime, run_log)

#                    elif 'waste' in data_str:
#                        global epoch_waste_dict
#                        job_name = data_str.split(' ')[0]
#                        epoch_waste_time = data_str.split(' ')[2]
#                        epoch_waste_dict[job_name] += int(epoch_waste_time)
#                    elif 'b_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_b[job].append(int(time.time() - b_start[job]))
#                        c_start[job] = time.time()
#                    elif 'c_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_c[job].append(int(time.time() - c_start[job]))
#                        d_start[job] = time.time()
#                    elif 'd_end' in data_str:
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        ovhd_d[job].append(int(time.time() - d_start[job]))
#                        ovhd_total[job].append(int(time.time() - ovhd_start[job]))
#                        if ovhd_start[job] != 0:
#                            overhead[job] += int(time.time() - ovhd_start[job])
#                            ovhd_start[job] = 0 
#                            if job in list(K80_job.values()):
#                                K80_start_time[job] = time.time()
#                            elif job in list(V100_job.values()):
#                                V100_start_time[job] = time.time()
#                                promote_start_time[job] = time.time()
#                    elif '1st_epoch' in data_str: # 'job50 1st_epoch 35'
#                        job_name = data_str.split(' ')[0]
#                        job = job_name.replace('job','')
#                        epoch_time = int(data_str.split(' ')[2])
#                        if job in list(K80_job.values()):
#                            k80_1st[job].append(epoch_time)
#                        elif job in list(V100_job.values()):
#                            v100_1st[job].append(epoch_time)
                    #if 'ckpt_qual' in data_str or 'finish' in data_str or 'checkpoint' in data_str:
                    #    print('received ' + data_str)
                    connection.sendall(b'success')
                    #time.sleep(5)
                else:
                    break
        finally:
            connection.close()

