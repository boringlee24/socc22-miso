import pdb
import json
import sys
import subprocess
import time
import os
home = os.environ.get('HOME')
import argparse
import re

# run this function at node level
def init_mig():
    for gpu in range(2):
        cmd = f'./enable_mig.sh {gpu}'
        p = subprocess.Popen([cmd], shell=True)
        p.wait()
   
def disable_mps():
    cmd = f'pkill -9 nvidia-cuda-mps'
    p = subprocess.Popen([cmd], shell=True)
    p.wait()

def disable_mig():
    for gpu in range(2):
        cmd = f'sudo nvidia-smi -i {gpu} -mig 0'
        p = subprocess.Popen([cmd], shell=True)
        p.wait()

def reset_mig(gpu):
    cmd = f'sudo nvidia-smi mig -i {gpu} -dci'
    # Note: need to make sure the reset is successful
    success = False
    while not success:
        p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
        p.wait()
        read = str(p.stdout.read())
        if 'Unable to destroy' not in read:
            success = True
        else:
            print('Trying again...')
            time.sleep(0.5)
    cmd = f'sudo nvidia-smi mig -i {gpu} -dgi'
    p = subprocess.Popen([cmd], shell=True)
    p.wait()

def create_ins(gpu, ins):
    id_map = {'1g.5gb': 19, '2g.10gb': 14, '3g.20gb': 9, '4g.20gb': 5, '7g.40gb': 0}
    ins_code = id_map[ins]
    cmd = f'sudo nvidia-smi mig -i {gpu} -cgi {ins_code}'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read())
    ID = re.findall(r'\d+', read)[0]
    # need to retrieve GPU instance ID from output
    cmd = f'sudo nvidia-smi mig -i {gpu} -gi {ID} -cci'
    p = subprocess.Popen([cmd], shell=True)
    p.wait()

def do_partition(gpu, partition): # partition is a list of slice # code is partition code, e.g. '0', '1',...
    id_map = {1: 19, 2: 14, 3: 9, 4: 5, 7: 0}
    ins_code = [str(id_map[k]) for k in partition]
    code_str = ','.join(ins_code)
    cmd = f'sudo nvidia-smi mig -i {gpu} -cgi {code_str}'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    cmd = f'sudo nvidia-smi mig -i {gpu} -cci'
    p = subprocess.Popen([cmd], shell=True)
    p.wait()
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulator')
    parser.add_argument('--init', action='store_true', help='initialize MIG mode', default=False)
    parser.add_argument('--disable', action='store_true', help='disable MIG mode', default=False)
    parser.add_argument('--disable_mps', action='store_true', help='disable MPS mode', default=False)    
    parser.add_argument('--reset', action='store_true', help='reset MIG slices', default=False)
    parser.add_argument('--create', action='store_true', help='create MIG slice', default=False)
    parser.add_argument('--gpu', type=int, help='select gpu', default=0)
    parser.add_argument('--instance', type=str, help='select instance to create', default='7g.40gb')
    parser.add_argument('--part', action='store_true', help='create partition', default=False)

    args = parser.parse_args()

    if args.init:
        init_mig()
    elif args.disable:
        disable_mig() 
    elif args.disable_mps:
        disable_mps()
    elif args.reset:
        reset_mig(args.gpu) 
    elif args.create:
        create_ins(args.gpu, args.instance)
    elif args.part:
        ts = time.time()
        reset_mig(args.gpu) 
        do_partition(args.gpu, [4,2,1])
        print(time.time() - ts)
