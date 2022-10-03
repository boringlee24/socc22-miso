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

class Oracle(Experiment):

    def __init__(self, args, physical_nodes):
        super().__init__(args, physical_nodes)
        self.avail_gpus = set(self.gpu_states[:])
        self.tc = 'oracle'

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False
        for instance in GPU_status.n2s_reverse.values():
            if instance in self.perf_pred[job%100]:
                min_size = instance
                break
        allowed_gpus = []
        for gpu in gpu_list:
            if gpu.max_allowed != 'full' and int(time.time()) - self.ckpt_buffer[gpu] > 3:
                if int(gpu.max_allowed.split('g.')[0]) >= int(min_size.split('g.')[0]):
                    allowed_gpus.append(gpu)
        # sort by number of active jobs
        sorted_gpus = sorted(allowed_gpus, key=lambda x: len(x.active_jobs), reverse=False)
        if len(sorted_gpus) > 0:
            gpu_sched = sorted_gpus[0]
            new_partition, new_jobs, new_code = gpu_sched.miso_optimize(job, self.perf_actual)
            sched_done = True
            gpuid = self.gpu_states.index(gpu_sched) # TODO: very important not to use gpu_list
            real_node, real_gpu = self.GPU_LUT(gpuid)
            # if this is the only job, then just start the job on it
            if len(new_jobs) == 1:
                if gpu_sched.encoded != new_code: 
                    config_gpu(real_node, real_gpu, new_code)
                start_job(real_node, job, real_gpu, new_jobs.index(job))
                gpu_sched.implement_miso_opt(new_partition, new_jobs, new_code, self.perf_actual)
                self.job_exe[job] = (gpuid, new_jobs.index(job))
                print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                print(f'job {job} scheduled on GPU {gpu_sched.index}, {real_node} device {real_gpu}, partition {new_partition}', file=run_log, flush=True)

            # otherwise, checkpoint current running jobs, if valid, resume them and start newly arrived job
            else:
                old_jobs = new_jobs[:]
                old_jobs.remove(job)
                valid = save_jobs(real_node, old_jobs, self, run_log)
                if not valid:
                    return False
                # reaching this step means checkpoint is done, now resume                
                time.sleep(1)
                # if new_code != gpu_sched.encoded:
                #     raise RuntimeError('try schedule did not re-partition GPU')
                if gpu_sched.encoded != new_code: 
                    config_gpu(real_node, real_gpu, new_code)
                print(f'Schedule time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                print(f'Start job {job}, GPU {gpu_sched.index} re-partitioned into {new_partition} job {new_jobs}, {real_node} device {real_gpu}', file=run_log, flush=True)

                for j in new_jobs:
                    if j == job:
                        start_job(real_node, j, real_gpu, new_jobs.index(j))
                        self.job_exe[j] = (gpuid, new_jobs.index(j))
                    else:
                        resume_batch = self.ckpt_batch[j]
                        resume_job(real_node, j, real_gpu, new_jobs.index(j), resume_batch)
                        self.job_exe[j] = (gpuid, new_jobs.index(j))
                        migration[j] += 1
                gpu_sched.implement_miso_opt(new_partition, new_jobs, new_code, self.perf_actual)
        return sched_done

    def run(self, args):
        run_log = open('logs/experiment_oracle.log','w')

        ####### start job listener ##########
        stop_event = Event()
        x = threading.Thread(target=thread_func, daemon=True, args=(stop_event, self, run_log, 'oracle'))
        x.start()

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
                sched_done = self.try_schedule(job, emptied_list, migration, run_log)
                if sched_done:
                    arrived_jobs.pop(0)
                    self.sched_time[job] = int(time.time())
                else: # stop scheduling jobs, follow a strict FIFO pattern
                    break

#            if no more arrived jobs can schedule, repartition emptied gpus:
            cnt_active = 0
            for gpu in self.gpu_states:
                if 'idle' in gpu.jobs and len(arrived_jobs) == 0:
                    real_node, real_gpu = self.GPU_LUT(self.gpu_states.index(gpu))
                    new_partition, new_jobs, new_code = gpu.miso_idle_optimize(self.perf_actual) # TODO: make this follow try schedule
                    # if currently no jobs, just reset the GPU and continue the loop
                    if new_jobs == ['idle']:
                        if gpu.encoded != new_code:
                            config_gpu(real_node, real_gpu, new_code)
                            print(f'Reset time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                            print(f'GPU {gpu.index} re-partitioned back to {new_partition} without job, {real_node} device {real_gpu}', file=run_log, flush=True)
                            gpu.implement_miso_opt(new_partition, new_jobs, new_code, self.perf_actual)
                        continue
                    # otherwise, checkpoint, reconfigure MIG, resume                      
                    valid = save_jobs(real_node, new_jobs, self, run_log)
                    if not valid:
                        continue
                    time.sleep(1)
                    if gpu.encoded == new_code:
                        raise RuntimeError('Check failed: GPU should not have the same partition as before')
                    config_gpu(real_node, real_gpu, new_code)
                    print(f'Re-partition time: {int(time.time()-self.start_time)}', file=run_log, flush=True)
                    print(f'GPU {gpu.index} re-partitioned into {new_partition} job {new_jobs}, {real_node} device {real_gpu}', file=run_log, flush=True)
                    for j in new_jobs:
                        resume_batch = self.ckpt_batch[j]
                        resume_job(real_node, j, real_gpu, new_jobs.index(j), resume_batch)
                        gpuid = self.gpu_states.index(gpu)
                        self.job_exe[j] = (gpuid, new_jobs.index(j))
                        migration[j] += 1
                    gpu.implement_miso_opt(new_partition, new_jobs, new_code, self.perf_actual)

                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in self.gpu_states]))
        
#            # sanity check
            for gpu in self.gpu_states:
                if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb' and len(arrived_jobs) == 0:
                    raise RuntimeError('Check failed: GPU should not have bubble')
        
            ################ check if termination condition is met ################
        
            if sum(self.finish.values()) == len(self.finish) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {int(time.time()-self.start_time)}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = int(time.time()-self.start_time)
                self.overall_rate.append(self.span_time)
                break
            elif int(time.time()-self.start_time) >= 18000:
                pdb.set_trace()
       
        ########################
        Path('logs/oracle').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.arrive_time[job]
            QT[job] = self.sched_time[job] - self.arrive_time[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        for metric, name in zip([JCT, JRT, QT], ['JCT', 'JRT', 'QT']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/oracle/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/oracle/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/oracle/completion.json', 'w') as f:
            json.dump(self.completion, f, indent=4)
        with open('logs/oracle/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/oracle/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/oracle/ckpt_dict.json', 'w') as f:
            json.dump(self.ckpt_dict, f, indent=4)
        with open('logs/oracle/ckpt_ovhd.json', 'w') as f:
            json.dump(self.ckpt_ovhd, f, indent=4)
        with open('logs/oracle/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)

        self.term_thread()
        stop_event.set()
#        print('trying to join threads')    
#        x.join()
        print('done')
