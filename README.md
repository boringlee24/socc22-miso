# MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters
## Published at 2022 ACM Symposium on Cloud Computing (SoCC '22)

<!-- ## Experiment Setup -->
This repository require access to NVIDIA A100 GPUs and sudo access to control the GPU.

Multi-Instance GPU user guide: [https://docs.nvidia.com/datacenter/tesla/mig-user-guide/](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)

## Preparation

#### Environment

Below are the software environment specifications:

OS: CentOS 7
Virtual Environment Manager: Anaconda 4.10.3
CUDA: 11.4
NVIDIA Driver: 470.82.01

Use the ``environment.yml`` file to install the virtual environment in Anaconda

```
conda env create -f environment.yml
```

then activate the environment in every node.

Make sure there this repo directory is ``/home/${USER}/GIT/socc22-miso`` where `${USER}` is the username. Also get a scratch directory available for temporary storage. Currently the scratch directory is `/scratch/${USER}`. If need to use another scratch directory, replace all `/scratch/${USER}` instances in this repo.

#### GPU Node Setup
First download the necessary files (e.g., datasets) needed for the workloads. Go to this Google drive link and [download](https://drive.google.com/file/d/1pcPcPNdDRSYTMnwuibjBSeobm1tGFmxE/view?usp=sharing) the file and unzip:
`unzip MISO_Workload.zip`

On each GPU node, first copy the necessary files into memory by modifying the files ``workloads/copy_memory.sh`` and ``workloads/clear_memory.sh``. Replace ''/dev/shm/tmp'' with the system shared memory location if not on Linux, and replace ''/work/li.baol/MISO_Workload/'' with the path where you extracted the .zip file. Then run

```
./workloads/clear_memory.sh
./workloads/copy_memory.sh
```

On each GPU node, do the following to set up MIG:

Run the following command to enable MIG

```
python mig_helper.py --init
```

Record the MIG slice UUID as lookup tables.

```
python export_cuda_device_auto.py
```

Wait for it to finish, then do the same above for the next GPU node. At this point, all GPUs have been set up and ready to go.

On each GPU, run the following command:

```
python gpu_server.py
```

## Start running

Allocate a CPU node as the scheduler, it should be able to communicate with the GPU nodes through TCP. 

Use 4 A100 GPUs to verify the code can work in your system. In the ``run.py`` script, find the variable "physical_nodes". In the current version, both items represent the hostname of each node, meaning two nodes each containing two GPUs. Modify this variable to match your system.

On the CPU (scheduler) node, run the following script:

```
python run.py --arrival 100 --num_gpu 4 --num_job 30 --random_trace
```

It will take several hours to finish these shortened experiments. If succesful, this means the repository has been successfully set up.

## Clean up

Disable MIG and MPS, clear up memory

```
python mig_helper.py --disable
./disable_mps.sh
./workloads/clear_memory.sh
```

## Note

You can reach me at my email: li.baol@northeastern.edu