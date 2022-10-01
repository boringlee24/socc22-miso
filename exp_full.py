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
from controller_helper import *
import threading
import _thread
sys.path.append(f'/home/{user}/GIT/mig_exp/workloads')
from send_signal import send_signal
import socket

class Experiment:
    def __init__(self, args, physical_nodes):
        self.node_list = physical_nodes
        self.start_time = 0 # real timer
        # shared attributes across different scheduling policies
        random.seed(args.seed)
        np.random.seed(args.seed+1)
        with open(f'/home/{user}/GIT/mig_exp/mps/scheduler/trace/trace_100.json') as f:
            job_dict = json.load(f)

        self.job_runtime = {} # job information
        if args.random_trace and args.num_job <= 100:            
            print('The used trace is randomly sampled from original trace')
            rand_ind = random.sample(range(len(job_dict)), args.num_job)
            for i in rand_ind:                
                self.job_runtime[i] = int(job_dict[str(i)])
        else:
            for i in range(args.num_job):
                index = i % 100
                self.job_runtime[i] = int(job_dict[str(index)]) # 0 to 199
        
        arrival_order = random.sample(list(self.job_runtime), len(self.job_runtime)) # job arrival order
        self.queue_dict = {} # contains job arrive time
        arrival_time = 0
        if args.flat_arrival:
            for job in arrival_order:
                arrival_time += np.random.poisson(args.arrival)
                self.queue_dict[job] = arrival_time
        else:            
            for job in arrival_order:
                rate = args.arrival / 2 if len(self.queue_dict) <= int(args.num_job/3) else args.arrival
                arrival_time += np.random.poisson(rate)
                self.queue_dict[job] = arrival_time

        self.filler_jobs = set()
        if args.filler:
            i, MAX = 0, args.num_gpu
            for job, arrival in self.queue_dict.items():
                if i < MAX:
                    self.queue_dict[job] = 0
                    self.filler_jobs.add(job)
                    i += 1
                elif i == MAX:
                    offset = arrival - args.arrival
                    i += 1
                    self.queue_dict[job] = arrival - offset
                else:
                    self.queue_dict[job] = arrival - offset
                    
        self.sched_time = {} # used to record queueing delay
        for job in self.job_runtime:
            self.sched_time[job] = 0
        self.comp_time = {} # used to record JCT and JRT
        for job in self.job_runtime:
            self.comp_time[job] = 0

        self.gpu_states = []
        self.emptied_gpu, self.ckpt_buffer = {}, {}

        for i in range(args.num_gpu):
            self.gpu_states.append(GPU_status(i))
        for i in self.gpu_states:
            self.ckpt_buffer[i] = 0

        self.completion, self.job_exe, self.pid_dict, self.ckpt_dict, \
        self.finish, self.arrive_time, self.ckpt_start, self.ckpt_ovhd, self.ckpt_batch \
        = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for j in self.job_runtime:
            self.completion[j] = 0
            self.job_exe[j] = (None, None)
            self.pid_dict[j] = 0
            self.ckpt_dict[j] = 0
            self.finish[j] = 0
            self.ckpt_start[j] = 0
            self.ckpt_ovhd[j] = []
            self.ckpt_batch[j] = 0
    
        with open(f'/home/{user}/GIT/mig_exp/mps/scheduler/simulator/job_models.json') as f:
            job_models = json.load(f)
        # map job model to speedup (predicted and actual)
        self.perf_actual, self.perf_pred = get_speedup(job_models, args.error_mean, args.error_std)

        self.tc = 'full'
        self.overall_rate = []
        self.span_time = 0

    def term_thread(self):
        # KILL THE THREAD
        message = f'term_thread'
        send_signal(socket.gethostname(), 10002, message)       
    
    def GPU_LUT(self, gpuid):
        gpu_per_node = 2
        quotient = gpuid // gpu_per_node
        remainder = gpuid % gpu_per_node
        real_node = self.node_list[quotient]
        real_gpu = str(remainder)
        return real_node, real_gpu

    def get_rate(self, gpu):
        deg = gpu.eval_degradation(self.perf_actual)
        rate = [1/k for k in deg]
        return round(sum(rate),3)

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False

        avail_gpu = [g for g in gpu_list if g.jobs == ['idle']]
        if len(avail_gpu) > 0:
            gpu = avail_gpu[0]
            gpu.jobs = [job]
            gpu.max_allowed = 'full'
            sched_done = True
            gpuid = self.gpu_states.index(gpu) # TODO: very important not to use gpu_list
            real_node, real_gpu = self.GPU_LUT(gpuid)
            start_job(real_node, job, real_gpu, 0)
            self.job_exe[job] = (gpuid, 0)
            print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu.index}, {real_node} device {real_gpu}', file=run_log, flush=True)
        return sched_done

    def run(self, args):
        run_log = open('logs/experiment_full.log','w')

        ####### start job listener ##########
        x = threading.Thread(target=thread_func, daemon=True, args=(self, run_log, 'full'))
        x.start()
        time.sleep(10)

        ####### initialize all GPUs #########
        for real_node in self.node_list:
            kill_all(real_node)
            broadcast_host(real_node, self)
        time.sleep(10)
        for gpu in self.gpu_states:
            real_node, real_gpu = self.GPU_LUT(gpu.index)
            config_gpu(real_node, real_gpu, 0)

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
                    sched_done = self.try_schedule(job, self.gpu_states, migration, run_log)
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
                if curr_time - emp_time >= 3: # give it 3 seconds to breath
                    emptied_list.append(gpu)
            # first see if jobs in arrived_jobs can be scheduled on emptied gpus
            for job in arrived_jobs[:]:
                sched_done = self.try_schedule(job, emptied_list, migration, run_log)
                if sched_done:
                    arrived_jobs.pop(0)
                    self.sched_time[job] = int(time.time())
                else: # stop scheduling jobs, follow a strict FIFO pattern
                    break
            # if no more arrived jobs can schedule, repartition emptied gpus:
            cnt_active = 0
            for gpu in self.gpu_states:
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in self.gpu_states]))

            # sanity check
            for gpu in self.gpu_states:
                if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb':
                    raise RuntimeError('Check failed: GPU should not have bubble')

            ################ check if termination condition is met ################
        
            if sum(self.finish.values()) == len(self.finish) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {int(time.time()-self.start_time)}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = int(time.time()-self.start_time)
                self.overall_rate.append(self.span_time)
                break
            elif int(time.time()-self.start_time) >= 36000:
                pdb.set_trace()

        ########################
        Path('logs/full').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.arrive_time[job]
            QT[job] = self.sched_time[job] - self.arrive_time[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        for metric, name in zip([JCT, JRT, QT], ['JCT', 'JRT', 'QT']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/full/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/full/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/full/completion.json', 'w') as f:
            json.dump(self.completion, f, indent=4)
        with open('logs/full/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/full/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/full/ckpt_dict.json', 'w') as f:
            json.dump(self.ckpt_dict, f, indent=4)
        with open('logs/full/ckpt_ovhd.json', 'w') as f:
            json.dump(self.ckpt_ovhd, f, indent=4)
        with open('logs/full/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)

        self.term_thread()
