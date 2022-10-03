import json
from pathlib import Path
import numpy as np
import pdb
from collections import Counter
import ast
import itertools
import copy
import os
user = os.environ.get('USER')
import random
import time

def get_speedup(job_models, error_mean, error_std):
    perf_actual = {} # {1: ['7g':1,'4g':0.9,'3g':0.8]}. normalized to max, from 7g, 4g to 1g. If none, put 0.
    perf_pred = {}
    mig_path = f'/home/{user}/GIT/socc22-miso/mps/models/logs/mig'
    mig_slices = ['7g.40gb', '4g.20gb', '3g.20gb', '2g.10gb', '1g.5gb']
    for i, model in job_models.items():
        job = int(i)
        perf_actual[job] = {}
        perf_pred[job] = {}
        perf_list = []
        for mig in mig_slices:
            filename = f'{mig_path}/{mig}_{model}.json'
            if Path(filename).is_file():
                with open(filename) as f:
                    lat = json.load(f)
                mean_lat = []
                for key,val in lat.items():
                    mean_lat += val 
                mean_lat = mean_lat[1:]
                mean_lat = round(np.mean(mean_lat),4)
                perf_list.append(mean_lat)
        perf_list = np.asarray(perf_list)
        perf_list = np.round(perf_list / perf_list[:3].max(), 4) # slice till 3 because unet model prediction is normalized this way
        # inject error now
        error_gen = np.random.normal(error_mean, error_std, len(perf_list))
        error_gen = np.where(np.random.random(len(error_gen)) > 0.5, -error_gen, error_gen)
        perf_error = perf_list + error_gen

        for index, perf in enumerate(perf_list):
            perf_actual[job][mig_slices[index]] = perf
        for index, perf in enumerate(perf_error):
            perf_pred[job][mig_slices[index]] = perf

    return perf_actual, perf_pred

class MPS_GPU_Status:    
    def __init__(self, node_index):
        self.jobs = []
        self.index = node_index        
    @property
    def active_jobs(self):
        return [j for j in self.jobs if j != 'idle']
    @property
    def full(self):
        if len(self.jobs) == 3:
            return True
        else:
            return False
    def eval_degradation(self, jrt_actual, coloc_penalty=0.15): # evaluate average degradation of jobs. generate jrt_actual in caller.
        return_list = []        
        for job in self.jobs:
            mapped_job = job % 100
            full_time = jrt_actual[mapped_job]['full']            
            mps_time = jrt_actual[mapped_job]['mps'] * (1 + (len(self.jobs)-1)*coloc_penalty)
            deg = full_time / mps_time
            return_list.append(deg)
        return return_list        

class GPU_status:
    num_to_str = {7: '7g.40gb', 4: '4g.20gb', 3: '3g.20gb', 2: '2g.10gb', 1: '1g.5gb'}
    n2s_reverse = {1: '1g.5gb', 2: '2g.10gb', 3: '3g.20gb', 4: '4g.20gb', 7: '7g.40gb'}

    with open(f'/home/{user}/GIT/socc22-miso/mps/scheduler/partition_code.json') as f:
        partition_code = json.load(f)

    def __init__(self, node_index):
        self.partition = ['7g.40gb'] # a list of MIG slices
        self.encoded = 0 # encoded slice config, 0 means [7g]
        self.jobs = ['idle'] # a list of jobs, each corresponds to one slice in order
        self.index = node_index
        self.max_allowed = '7g.40gb' # max allowed new job slice type
    @property
    def active_jobs(self):
        return [j for j in self.jobs if j != 'idle']
    @property
    def inactive_slices(self):
        return [instance for instance, job in zip(self.partition, self.jobs) if job == 'idle']
    @property
    def max_inactive_slice(self):
        max_size = None
        max_slice_ind = None
        for ind, instance in enumerate(self.partition):
            code = int(instance.split('g.')[0])
            job = self.jobs[ind]
            if job == 'idle':
                if max_size is None:
                    max_size = code
                    max_slice_ind = ind
                else:
                    if code > max_size:
                        max_size = code
                        max_slice_ind = ind
        return max_size, max_slice_ind                    
    
    def get_job_slice(self, job):
        index = self.jobs.index(job)
        return self.partition[index]

    def check(self):
        if len(self.jobs) != len(self.partition):
            return False
        else:
            return True

    '''
    this is for static partition scheme, set max_allowed property
    '''
    def static_max_slice(self):
        new_max_slice = self.max_inactive_slice[0]
        if new_max_slice is None:
            self.max_allowed = 'full'
        else:
            self.max_allowed = GPU_status.num_to_str[new_max_slice]

    '''
    this is for static partition scheme, when there are idle slices for promotion
    '''
    def static_idle_optimize(self):
        inactive_index = {} # {index: size}
        active_index = {}
        for ind, instance in enumerate(self.partition):
            if self.jobs[ind] == 'idle':
                inactive_index[ind] = int(instance.split('g.')[0])
            else:
                active_index[ind] = int(instance.split('g.')[0])
        inactive_sorted = dict(sorted(inactive_index.items(), key=lambda x: x[1], reverse=True)) # from big to small
        active_sorted = dict(sorted(active_index.items(), key=lambda x: x[1])) # small to big
        migrated_jobs = []
        for i in range(min(len(inactive_sorted), len(active_sorted))):
            key1, val1 = list(inactive_sorted.keys())[i], list(inactive_sorted.values())[i]
            key2, val2 = list(active_sorted.keys())[i], list(active_sorted.values())[i]
            if val1 > val2:
                mig_job = self.jobs[key2]
                self.jobs[key1] = mig_job
                self.jobs[key2] = 'idle'
                migrated_jobs.append(mig_job)
            else:
                break
        self.static_max_slice()
        return migrated_jobs

    def static_idle_optimize_V2(self): 
        inactive_index = {} # {index: size}
        active_index = {}
        for ind, instance in enumerate(self.partition):
            if self.jobs[ind] == 'idle':
                inactive_index[ind] = int(instance.split('g.')[0])
            else:
                active_index[ind] = int(instance.split('g.')[0])
        inactive_sorted = dict(sorted(inactive_index.items(), key=lambda x: x[1], reverse=True)) # from big to small
        active_sorted = dict(sorted(active_index.items(), key=lambda x: x[1])) # small to big
        migrated_jobs = [] # [(job, original slice index, new slice index)]
        for i in range(min(len(inactive_sorted), len(active_sorted))):
            key1, val1 = list(inactive_sorted.keys())[i], list(inactive_sorted.values())[i]
            key2, val2 = list(active_sorted.keys())[i], list(active_sorted.values())[i]
            if val1 > val2:
                mig_job = self.jobs[key2]
                migrated_jobs.append((mig_job, key2, key1))
            else:
                break
        return migrated_jobs
    def update_static_migration(self, migrated_jobs):
        for pair in migrated_jobs:
            mig_job, key2, key1 = pair
            self.jobs[key1] = mig_job
            self.jobs[key2] = 'idle'
        self.static_max_slice()
           
    '''
    set modified object in-place
    '''
    def update_max_allowed(self, perf_dict):
        # get minimum slice size of each existing job
        min_slice = []
        for j in self.active_jobs:
            for code, slice_size in self.n2s_reverse.items():
                if slice_size in perf_dict[j % 100]:
                    min_slice.append(code)
                    break
        count = Counter(min_slice)
#        pdb.set_trace()
        for code in self.num_to_str.keys():
            new_count = copy.deepcopy(count)
            if code in new_count:
                new_count[code] += 1
            else:
                new_count[code] = 1
            # compare against all possible partition codes
            for partition in self.partition_code.values(): # [4,2,1]
                partition_cnt = Counter(partition) # {4:1, 2:1, 1:1}
                feasible = False
                for key, val in new_count.items():
                    if key in partition_cnt:
                        if partition_cnt[key] >= val:
                            feasible = True
                        else:
                            feasible = False
                            break
                    else:
                        feasible = False
                        break
                # if reach this step, it means this partition is viable
                # so update self.max_allowed
                if feasible == True:
                    self.max_allowed = self.num_to_str[code]
                    return 
        # if still no return, it means no more jobs are allowed here
        self.max_allowed = 'full'
        return
        
    '''
    given an encoded slice list, partition the GPU accordingly
    set all slices to idle
    '''
    def implement(self, slice_code, slice_list): # [3,1,1,1]
        self.partition = []
        self.jobs = []
        for mig_slice in slice_list:
            self.partition.append(self.num_to_str[mig_slice])
            self.jobs.append('idle')
        self.encoded = slice_code
    '''
    evaluate average degradation of jobs assigned to this GPU
    output: a list of degradations
    '''
    def eval_degradation(self, perf_actual): # evaluate average degradation of jobs
        return_list = []
        for mig, job in zip(self.partition, self.jobs):
            if job != 'idle':
                mapped_job = job % 100 # 100 jobs are duplicated multiple times
                full_time = perf_actual[mapped_job]['7g.40gb']
                mig_time = perf_actual[mapped_job][mig]
                deg = mig_time / full_time
                return_list.append(deg)
        return return_list

    def get_num_migrate(self, new_partition, new_assign):
        # make sure the new assign has the same jobs

        if set(self.active_jobs) != set(new_assign):
            raise RuntimeError('the new assignment is invalid')
        num_mig = 0
        migrated = []
        mig_partition = [int(k.split('g.')[0]) for k in self.partition]
#        pdb.set_trace()
        for j in self.active_jobs:
            ind = self.jobs.index(j)
            ind_new = new_assign.index(j)
            # if not same partition, must migration
            # otherwise if not same starting position, must migration
            if mig_partition[ind] != new_partition[ind_new]:
                num_mig += 1
                migrated.append(j)
                continue
            else:
                position_old = sum(mig_partition[:ind])
                position_new = sum(new_partition[:ind_new])
                if position_old != position_new:
                    num_mig += 1
                    migrated.append(j)
        if len(migrated) != num_mig:
            raise RuntimeError('error in get_num_migrate function')
        return num_mig, migrated
            
    '''
    reconfigure the partition and assignment exhaustively
    find best configuration that requires the least number of migrations
    '''
    def idle_partition_optimize(self, perf_dict):
        num_jobs = len(self.active_jobs)
        # if no jobs left, reset the GPU
        if num_jobs == 0:
            self.__init__(self.index)
            return 0, []
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition, best_migrated = [], [], []
        least_deg = 1000 # a large number, minimum is 1
        min_mig = 7
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(self.active_jobs):

                code = int(code_str)
                all_assigns = itertools.permutations(self.active_jobs[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    deg_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True
                    num_migrate, migrated = self.get_num_migrate(partition, assign) 
                    if num_migrate > min_mig:
                        continue

                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            deg_list.append(perf_dict[job][slice_type] / perf_dict[job]['7g.40gb'])
                    if not feasbile: # this assignment does not work
                        continue
                    mean_deg = np.mean(deg_list)
                    if num_migrate < min_mig: # degradation does not matter here 
                        least_deg, best_code, min_mig = mean_deg, code, num_migrate
                        best_assign = list(assign)
                        best_partition = partition[:]
                        best_migrated = migrated[:]
                    elif mean_deg < least_deg: # same migration, compare degradation
                        least_deg, best_code = mean_deg, code
                        best_assign = list(assign)
                        best_partition = partition[:]
                        best_migrated = migrated[:]
        if len(best_assign) == 0:
            raise RuntimeError('Error: idle partition optimize did not work')
        else:
            # modify the object
            self.partition = [self.num_to_str[p] for p in best_partition]
            self.encoded = best_code
            self.jobs = best_assign[:]
            self.update_max_allowed(perf_dict)
            return min_mig, best_migrated

    '''
    reconfigure the partition and assignment exhaustively
    find best configuration, no care to number of migration
    '''
    def idle_partition_optimize_V2(self, perf_dict, rand_gen=False):
        num_jobs = len(self.active_jobs)
        # if no jobs left, reset the GPU
        if num_jobs == 0:
            self.__init__(self.index)
            return 0, []
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition, best_migrated = [], [], []
        least_deg = 1000 # a large number, minimum is 1
        min_mig = 7
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(self.active_jobs):

                code = int(code_str)
                all_assigns = itertools.permutations(self.active_jobs[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    deg_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True
                    num_migrate, migrated = self.get_num_migrate(partition, assign) 
                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            deg_list.append(perf_dict[job][slice_type] / perf_dict[job]['7g.40gb'])
                    if not feasbile: # this assignment does not work
                        continue
                    if rand_gen:
                        mean_deg = random.randint(0,100)
                    else:
                        mean_deg = np.mean(deg_list)
                    if mean_deg < least_deg: # degradation does not matter here 
                        least_deg, best_code, min_mig = mean_deg, code, num_migrate
                        best_assign = list(assign)
                        best_partition = partition[:]
                        best_migrated = migrated[:]
        if len(best_assign) == 0:
            raise RuntimeError('Error: idle partition optimize did not work')
        else:
            # modify the object
            self.partition = [self.num_to_str[p] for p in best_partition]
            self.encoded = best_code
            self.jobs = best_assign[:]
            self.update_max_allowed(perf_dict)
            return min_mig, best_migrated

    '''
    Run this when there are idle MIG slices
    This requires re-partitioning the GPU to remove idle slices (bubble)
    '''
    def miso_idle_optimize(self, perf_dict):
        num_jobs = len(self.active_jobs)
        # if no jobs left, reset the GPU
        if num_jobs == 0:
            return [7], ['idle'], 0  
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition, best_migrated = [], [], []
        least_deg = 1000 # a large number, minimum is 1
        min_mig = 7
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(self.active_jobs):

                code = int(code_str)
                all_assigns = itertools.permutations(self.active_jobs[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    deg_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True
                    num_migrate, migrated = self.get_num_migrate(partition, assign) 
                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            deg_list.append(perf_dict[job][slice_type] / perf_dict[job]['7g.40gb'])
                    if not feasbile: # this assignment does not work
                        continue
                    mean_deg = np.mean(deg_list)
                    if mean_deg < least_deg: # degradation does not matter here 
                        least_deg, best_code, min_mig = mean_deg, code, num_migrate
                        best_assign = list(assign)
                        best_partition = partition[:]
                        best_migrated = migrated[:]
        if len(best_assign) == 0:
            raise RuntimeError('Error: idle partition optimize did not work')
        else:
            return best_partition, best_assign[:], best_code
#            # modify the object
#            self.partition = [self.num_to_str[p] for p in best_partition]
#            self.encoded = best_code
#            self.jobs = best_assign[:]
#            self.update_max_allowed(perf_dict)
#            return min_mig, best_migrated

    '''
    Exhaustively explore all configs that can support this many jobs (including current jobs)
        the input is a single job (sched 1 job at a time)
        accounts for out-of-memory on smaller slices
        modify the gpu_status in-place
    '''
    def single_gpu_optimize(self, job, perf_dict, return_STP=False):
        job_list = self.active_jobs[:] 
        job_list.append(job)
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition = [], []
        least_deg = 1000 # a large number, minimum is 1
        best_STP = 0
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(job_list):

                code = int(code_str)
                all_assigns = itertools.permutations(job_list[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    deg_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True

                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            deg_list.append(perf_dict[job][slice_type] / perf_dict[job]['7g.40gb'])
                    if not feasbile: # this assignment does not work
                        continue
                    mean_deg = np.mean(deg_list)
                    if mean_deg < least_deg:
                        least_deg, best_code = mean_deg, code
                        best_assign = list(assign)
                        best_partition = partition[:]
                        best_STP = sum([1/k for k in deg_list])
        if len(best_assign) == 0:
            if return_STP:
                return best_STP
            else:
                raise RuntimeError('Error: no assignment available for scheduled job')
        else:
            # modify the object
            self.partition = [self.num_to_str[p] for p in best_partition]
            self.encoded = best_code
            self.jobs = best_assign[:]
            self.update_max_allowed(perf_dict)
            if return_STP:
                return best_STP
            else:
                return least_deg

    '''
    optimize STP instead of mean time degradation
    '''
    def single_gpu_optimize_STP(self, perf_dict):
        job_list = self.active_jobs[:]         
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition = [], []
        best_STP = 0
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(job_list):

                code = int(code_str)
                all_assigns = itertools.permutations(job_list[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    stp_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True

                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            stp_list.append(perf_dict[job]['7g.40gb'] / perf_dict[job][slice_type])
                    if not feasbile: # this assignment does not work
                        continue
                    sum_stp = sum(stp_list)
                    if sum_stp > best_STP:
                        best_STP, best_code = sum_stp, code
                        best_assign = list(assign)
                        best_partition = partition[:]                
        # modify the object
        self.partition = [self.num_to_str[p] for p in best_partition]
        self.encoded = best_code
        self.jobs = best_assign[:]
        self.update_max_allowed(perf_dict)            
        return best_STP

    def miso_optimize(self, job, perf_dict):
        job_list = self.active_jobs[:] 
        job_list.append(job)
        # exhaustively search through all slices, and assignments
        best_code = 0
        best_assign, best_partition = [], []
        least_deg = 1000 # a large number, minimum is 1
        for code_str, partition in self.partition_code.items():
            if len(partition) == len(job_list):

                code = int(code_str)
                all_assigns = itertools.permutations(job_list[:])
                
                for assign in all_assigns: # [100, 20, 3]
                    deg_list = [] # degradation of all jobs in this assignment
                    # check if assignment is feasible
                    feasbile = True

                    for job_unmapped, instance in zip(assign, partition):
                        slice_type = self.num_to_str[instance]
                        job = job_unmapped % 100
                        if slice_type not in perf_dict[job]:
                            feasbile = False
                            break
                        else:
                            deg_list.append(perf_dict[job][slice_type] / perf_dict[job]['7g.40gb'])
                    if not feasbile: # this assignment does not work
                        continue
                    mean_deg = np.mean(deg_list)
                    if mean_deg < least_deg:
                        least_deg, best_code = mean_deg, code
                        best_assign = list(assign)
                        best_partition = partition[:]
        if len(best_assign) == 0:
            raise RuntimeError('Error: no assignment available for scheduled job')
        else:
            # do not modify the object yet
            return best_partition, best_assign[:], best_code

    def implement_miso_opt(self, best_partition, best_assign, best_code, perf_dict):
        self.partition = [self.num_to_str[p] for p in best_partition]
        self.encoded = best_code
        self.jobs = best_assign
        self.update_max_allowed(perf_dict)

'''
input: list of GPU status instances
output: dict of config, e.g., {7: 6, 4: 0, 3: 1, 2: 0, 1: 10}
'''
def get_mapped_config(gpu_list): # list of GPU_status
    full_list = []
    for gpu in gpu_list:
        full_list += gpu.partition
    return dict(Counter(full_list))
  
'''
input: dict of config in mapped space (e.g., {7: 6, 4: 1, 3: 1, 2: 2, 1: 3})
output: actual partition of every GPU, represented as list of GPU_status, modified in-place, non-return
'''
def implement_mapped_config(config_dict, gpu_list):
    num_gpu = len(gpu_list)
    with open(f'/home/{user}/GIT/socc22-miso/mps/scheduler/mapped_{num_gpu}gpu.json') as f:
        mapping = json.load(f)
    actual_config = ast.literal_eval(mapping[str(config_dict)]) # [0,1,1,2,12]
    with open(f'/home/{user}/GIT/socc22-miso/mps/scheduler/partition_code.json') as f:
        partition_code = json.load(f)
    for index, config in enumerate(actual_config):
        slice_list = partition_code[str(config)]
        gpu_list[index].implement(str(config), slice_list)
'''
evaluate a particular partition + assignment configuration (average degradation of all jobs)
'''
def eval_config(gpu_list, perf_dict):
    eval_list = []
    for gpu in gpu_list:
        eval_list += gpu.eval_degradation
    return np.mean(eval_list)
'''
create bidding for each slice type, given the current job list
output: {'7g.40gb': {1: 1.2, 2: 1.5}, '4g.20gb': ...}
'''
def create_bidding(job_list, perf_dict):
    bidding_dict = {}
    slice_type = ['7g.40gb', '4g.20gb', '3g.20gb', '2g.10gb', '1g.5gb']
    for mig_ind, mig in enumerate(slice_type[:-1]):
        bidding_dict[mig] = {}
        # unfinished, need to consider out-of-memory on smaller slices

'''
input: a list of GPU_status objects, partitioned, but all slices are idle,
       a list of jobs,
       job performances
modifies the objects in-place, writing its 'jobs' attribute
it is aware of MIG slice out-of-memory constraints of each job
output: boolean indicating whether this assignment supports all jobs
'''
def job_assignment(gpu_list, job_list, perf_dict):
    # make sure there are enough number of slices
    num_slices = 0
    for gpu in gpu_list:
        num_slices += len(gpu.partition)
    if num_slices < len(job_list):
        raise RuntimeError('number of slices must be enough to host all jobs')

    # start the bidding process
    slice_type = ['7g.40gb', '4g.20gb', '3g.20gb', '2g.10gb', '1g.5gb']
    for mig_ind, mig in enumerate(slice_type[:-1]):
        for gpu in gpu_list:
            for index, partition in enumerate(gpu.partition):
                if partition == mig:
                    # open the bid
                    bidding_dict = {}
                    if len(job_list) > 0:
                        for job in job_list:
                            mapped_job = job % 100
                            smaller_slice = slice_type[mig_ind+1]
                            if smaller_slice not in perf_dict[mapped_job]:
                                bidding_dict
                            bigger_time = perf_dict[mapped_job][partition] # time of bigger slice
                            smaller_time = perf_dict[mapped_job][smaller_slice]
                            bidding_dict[job] = smaller_time / bigger_time
                        winner = max(bidding_dict, key = bidding_dict.get) # job with highest bid
                        job_list.remove(winner)
                        gpu.jobs[index] = winner
                    else:
                        break
    # assign the rest of the job_list to 1g.5gb
    for gpu in gpu_list:
        for index, partition in enumerate(gpu.partition):
            if partition == '1g.5gb':
                if len(job_list) > 0:
                    gpu.jobs[index] = job_list[0]
                    job_list.pop(0)
                else:
                    break
    if len(job_list) > 0:
        return False
    else:
        return True
       
                        









