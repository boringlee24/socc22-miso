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
sys.path.append(f'/home/{user}/GIT/socc22-miso/mps/scheduler/simulator/')
from utils import *
import copy
from controller_helper import *
import threading
import _thread
from exp_full import Experiment
sys.path.append(f'/home/{user}/GIT/socc22-miso/workloads')
from send_signal import send_signal
import socket
from threading import Event

class MPS(Experiment):

    def __init__(self, args, physical_nodes):
        super().__init__(args, physical_nodes)
        self.tc = 'mps'
        self.gpu_states = []
        for i in range(args.num_gpu):
            self.gpu_states.append(MPS_GPU_Status(i))        

    # try to schedule job on a list of GPUs
    def try_schedule(self, job, gpu_list, migration, run_log, mps_lvl):
        sched_done = False
        
        avail_gpu = [g for g in gpu_list if not g.full]
        if len(avail_gpu) > 0:
            gpu = avail_gpu[0]
            gpu.jobs.append(job)
            sched_done = True
            gpuid = self.gpu_states.index(gpu)
            real_node, real_gpu = self.GPU_LUT(gpuid)
            mps_start(real_node, job, real_gpu, level=mps_lvl)
            self.job_exe[job] = (gpuid, 0)
            print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu.index}, {real_node} device {real_gpu}', file=run_log, flush=True)
        return sched_done            

    def run(self, args, mps_lvl=33): 
        run_log = open('logs/experiment_mps.log','w')

        ####### start job listener ##########
        stop_event = Event()
        x = threading.Thread(target=thread_func, daemon=True, args=(stop_event, self, run_log, 'mps'))
        x.start()

        ####### initialize all GPUs #########
        for real_node in self.node_list:
            kill_all(real_node)
            broadcast_host(real_node, self)
        time.sleep(10)
        for gpu in self.gpu_states:
            real_node, real_gpu = self.GPU_LUT(gpu.index)
            start_mps(real_node, real_gpu)
            # config_gpu(real_node, real_gpu, 0)

        ###### initialize some variables ########
        queue = list(self.queue_dict)
        queue_ind = 0
    
        active_jobs_per_gpu = [] # time series of total number of jobs running
        arrived_jobs = []
        progress = {}
        migration  = {}
        for j in self.job_runtime:
            migration[j] = 0
        time.sleep(10)
        
        ##### start running ###########
        self.start_time = int(time.time())
        progress_time = int(time.time())

        while True:
            passed_time = int(time.time() - self.start_time)
#            if passed_time >= 900:
#                pdb.set_trace()
            while queue_ind < len(queue) and self.queue_dict[queue[queue_ind]] <= passed_time:
                arrived_jobs.append(queue[queue_ind])
                self.arrive_time[queue[queue_ind]] = int(time.time())
                queue_ind += 1

            if len(arrived_jobs) >= 1:
                '''
                priority
                1. there is idle GPU, so no migration at all
                2. least number of current jobs currently running, randomly pick one, order does not matter
                '''
                for job in arrived_jobs[:]:
                    sched_done = self.try_schedule(job, self.gpu_states, migration, run_log, mps_lvl)
                    if sched_done:
                        arrived_jobs.pop(0)
                        self.sched_time[job] = int(time.time())
                    else: # stop scheduling jobs, follow a strict FIFO pattern
                        break

            ############### wait for next iteration, job is running ##########
        
            self.emptied_gpu = {}
            time.sleep(args.step)            

            if int(time.time() - progress_time) >= 60:
                progress[int(time.time() - self.start_time)] = sum(list(self.completion.values()))
                progress_time = int(time.time())
             
            curr_time = int(time.time())
            emptied_list = []
            for gpu, emp_time in self.emptied_gpu.items():
                if curr_time - emp_time > 3: # give it 3 seconds to breath
                    emptied_list.append(gpu)
       
#            # first see if jobs in arrived_jobs can be scheduled on emptied gpus
            for job in arrived_jobs[:]:
                sched_done = self.try_schedule(job, emptied_list, migration, run_log, mps_lvl)
                if sched_done:
                    arrived_jobs.pop(0)
                    self.sched_time[job] = int(time.time())
                else: # stop scheduling jobs, follow a strict FIFO pattern
                    break

#            if no more arrived jobs can schedule, repartition emptied gpus:
            cnt_active = 0
            for gpu in self.gpu_states:
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            # MPS jobs cannot calculate overall rate like this
            # self.overall_rate.append(sum([self.get_rate(gpu) for gpu in self.gpu_states]))
        
#            # sanity check
            for gpu in self.gpu_states:
                if len(gpu.jobs) > 3:
                    raise RuntimeError('Check failed: GPU should not have >3 jobs')
            
            ################ check if termination condition is met ################
        
            if sum(self.finish.values()) == len(self.finish) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {int(time.time()-self.start_time)}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = int(time.time()-self.start_time)
                self.overall_rate.append(self.span_time)
                break
            elif int(time.time()-self.start_time) >= 36000:
                pdb.set_trace()
       
        ########################
        Path('logs/mps').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.arrive_time[job]
            QT[job] = self.sched_time[job] - self.arrive_time[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        for metric, name in zip([JCT, JRT, QT], ['JCT', 'JRT', 'QT']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/mps/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/mps/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/mps/completion.json', 'w') as f:
            json.dump(self.completion, f, indent=4)
        with open('logs/mps/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/mps/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/mps/ckpt_dict.json', 'w') as f:
            json.dump(self.ckpt_dict, f, indent=4)
        with open('logs/mps/ckpt_ovhd.json', 'w') as f:
            json.dump(self.ckpt_ovhd, f, indent=4)
        with open('logs/mps/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)

        self.term_thread()
        stop_event.set()
#        print('trying to join threads')    
#        x.join()
        print('done')
