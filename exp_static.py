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

class Static(Experiment):

    def __init__(self, args, physical_nodes):
        super().__init__(args, physical_nodes)
        self.tc = 'static'

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False

        for instance in GPU_status.n2s_reverse.values():
            if instance in self.perf_actual[job%100]:
                min_size = instance
                break
        allowed_slices = []
        for code, instance in GPU_status.num_to_str.items():
            if code >= int(min_size.split('g.')[0]):
                allowed_slices.append(instance)

        gpu_avail = [g for g in gpu_list if g.max_allowed != 'full']
        gpu_avail = [g for g in gpu_avail if int(time.time())-self.ckpt_buffer[g]>3]

        # sort by maximum idle slice size
        sorted_gpus = sorted(gpu_avail, key=lambda x: x.max_inactive_slice[0], reverse=True)

        if len(sorted_gpus) > 0:
            gpu = sorted_gpus[0]
            slice_code, slice_ind = gpu.max_inactive_slice
            if GPU_status.num_to_str[slice_code] in allowed_slices:
                gpu.jobs[slice_ind] = job
                gpu.static_max_slice()
                sched_done = True
                gpuid = self.gpu_states.index(gpu) # TODO: very important not to use gpu_list
                real_node, real_gpu = self.GPU_LUT(gpuid)
                start_job(real_node, job, real_gpu, slice_ind)
                self.job_exe[job] = (gpuid, slice_ind)
                print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                print(f'job {job} scheduled on GPU {gpu.index}, {real_node} device {real_gpu}, partition {gpu.partition}, jobs {gpu.jobs}', file=run_log, flush=True)
        return sched_done

    def run(self, args, slice_code=6, partition=[3,2,2]):
        run_log = open('logs/experiment_static.log','w')

        ####### start job listener ##########
        stop_event = Event()
        x = threading.Thread(target=thread_func, daemon=True, args=(stop_event, self, run_log, 'static'))
        x.start()

        ####### initialize all GPUs #########
        for real_node in self.node_list:
            kill_all(real_node)
            broadcast_host(real_node, self)
        time.sleep(10)
        for gpu in self.gpu_states:
            real_node, real_gpu = self.GPU_LUT(gpu.index)
            config_gpu(real_node, real_gpu, slice_code)
            gpu.implement(slice_code, partition)
            gpu.max_allowed = GPU_status.num_to_str[max(partition)]

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
                    sched_done = self.try_schedule(job, self.gpu_states, migration, run_log)
                    if sched_done:
                        arrived_jobs.remove(job)
                        self.sched_time[job] = int(time.time())
           
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
                sched_done = self.try_schedule(job, emptied_list, migration, run_log)
                if sched_done:
                    arrived_jobs.remove(job)
                    self.sched_time[job] = int(time.time())
#            if no more arrived jobs can schedule, promote within current jobs:
            cnt_active = 0
            for gpu in self.gpu_states:
                if 'idle' in gpu.jobs:
                    migrated_jobs = gpu.static_idle_optimize_V2()
                    if len(migrated_jobs) > 0:
                        mig_job_list = [j[0] for j in migrated_jobs]
                        real_node, real_gpu = self.GPU_LUT(self.gpu_states.index(gpu))
                        valid = save_jobs(real_node, mig_job_list, self, run_log)
                        if not valid:
                            continue
                        # reaching this step means checkpoint is done, now resume
                        time.sleep(1)
                        gpu.update_static_migration(migrated_jobs)
                        for j in migrated_jobs:
                            jobid, orig_slice, new_slice = j
                            migration[jobid] += 1
                            resume_batch = self.ckpt_batch[jobid]
                            resume_job(real_node, jobid, real_gpu, new_slice, resume_batch)
                            self.job_exe[jobid] = (gpu.index, new_slice)
                        print(f'Restart time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                        print(f'Promotion on GPU {gpu.index}, jobs {gpu.jobs}, slice {gpu.partition}', file=run_log, flush=True)

                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in self.gpu_states]))
        
            ################ check if termination condition is met ################
        
            if sum(self.finish.values()) == len(self.finish) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {int(time.time()-self.start_time)}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = int(time.time()-self.start_time)
                self.overall_rate.append(self.span_time)
                break
#            elif int(time.time()-self.start_time) >= 18000:
#                pdb.set_trace()
       
        ########################
        Path('logs/static').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.arrive_time[job]
            QT[job] = self.sched_time[job] - self.arrive_time[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        for metric, name in zip([JCT, JRT, QT], ['JCT', 'JRT', 'QT']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/static/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/static/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/static/completion.json', 'w') as f:
            json.dump(self.completion, f, indent=4)
        with open('logs/static/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/static/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/static/ckpt_dict.json', 'w') as f:
            json.dump(self.ckpt_dict, f, indent=4)
        with open('logs/static/ckpt_ovhd.json', 'w') as f:
            json.dump(self.ckpt_ovhd, f, indent=4)
        with open('logs/static/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)

        self.term_thread()
        stop_event.set()
#        print('trying to join threads')    
#        x.join()
        print('done')
