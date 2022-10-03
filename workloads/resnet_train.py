"""
#Trains a ResNet on the CIFAR10 dataset.

"""

from __future__ import print_function
from tensorflow import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50#, ResNet101, ResNet152
from keras import models, layers, optimizers
from datetime import datetime
import tensorflow as tf
import socket
import numpy as np
import os
import pdb
import sys
import argparse
import time
import json 
from pathlib import Path
import signal
import subprocess
import grpc
home = os.environ.get('HOME')
user = os.environ.get('USER')
hostname = socket.gethostname()
sys.path.append(f'{home}/GIT/socc22-miso/mps/grpc')
import grpc_pb2, grpc_pb2_grpc
import pdb
import glob
from send_signal import send_signal
from checkpoint_helper import CustomCheckpoint

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model', metavar='MODEL', type=str, help='specific model name', default='resnet50')
parser.add_argument('--lr', metavar='LEARNING_RATE', type=float, help='learning rate', default=0.001)
parser.add_argument('--job_id', metavar='JOB', type=str, help='ID of job from trace', default='0')
parser.add_argument('--partition', metavar='par', type=str, help='thread partition percentage', default='50')
parser.add_argument('--mps_set', action='store_true', help='enable this if operating in MPS mode', default=False)
parser.add_argument('--port', metavar='pt', type=int, help='port for grpc', default=50051)
parser.add_argument('--mps_sync', action='store_true', help='enable this if requires MPS sync start', default=False)
parser.add_argument('--iters', type=int, help='iterations to train', default=200)
parser.add_argument('--no_ckpt', action='store_true', help='no checkpoint at end of each epoch', default=False)
parser.add_argument('--resume', action='store_true', help='resume from previous checkpoint', default=False)
parser.add_argument('--node', type=str, help='node of host (scheduler)')
parser.add_argument('--start_batch', type=int, help='starting batch from resume', default=0)
parser.add_argument('--cuda_device', type=str, help='cuda device when running mps', default='0')

args = parser.parse_args()
save_file = f'/scratch/{user}/mig_ckpt'
starting_epoch = 0

# first step is to update the PID
pid = os.getpid()
message = f'job {args.job_id} pid {pid}' 
send_signal(args.node, 10002, message)

if args.resume:
    if len(glob.glob(f'{save_file}/job{args.job_id}_epoch*.h5')) > 0:
        saved = glob.glob(f'{save_file}/job{args.job_id}_epoch*.h5')[0]
        starting_epoch = int(saved.split('_epoch')[1].split('.')[0]) + 1

if args.mps_set: 
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition
    os.environ['CUDA_MPS_LOG_DIRECTORY']=f'/scratch/{user}/mps_log/nvidia-log-{hostname}/{args.cuda_device}'
    os.environ['CUDA_MPS_PIPE_DIRECTORY']=f'/scratch/{user}/mps_log/nvidia-mps-{hostname}/{args.cuda_device}'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Training parameters
batch_size = args.batch_size  # orig paper trained all networks with batch_size=128
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

n = 3

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = args.lr #1e-3
    print('Learning rate: ', lr)
    return lr

#model = models.Sequential()
#
#if '50' in args.model:
#    base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)
#elif '101' in args.model:
#    base_model = ResNet101(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)
#elif '152' in args.model:
#    base_model = ResNet152(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)
#
##base_model.summary()
#
##model.add(layers.UpSampling2D((2,2)))
##model.add(layers.UpSampling2D((2,2)))
##model.add(layers.UpSampling2D((2,2)))
#model.add(base_model)
#model.add(layers.Flatten())
##model.add(layers.BatchNormalization())
##model.add(layers.Dense(128, activation='relu'))
##model.add(layers.Dropout(0.5))
##model.add(layers.BatchNormalization())
##model.add(layers.Dense(64, activation='relu'))
##model.add(layers.Dropout(0.5))
##model.add(layers.BatchNormalization())
#model.add(layers.Dense(10, activation='softmax'))#, kernel_initializer='he_uniform'))
#
#model.compile(loss='categorical_crossentropy',
#              optimizer=Adam(lr=lr_schedule(0)),
#              metrics=['accuracy'])
#
#
##model.fit(x_train, y_train, epochs=1, batch_size=20, validation_data=(x_test, y_test))
#
##def get_flops(model):
##    run_meta = tf.RunMetadata()
##    opts = tf.profiler.ProfileOptionBuilder.float_operation()
##
##    # We use the Keras session graph in the call to the profiler.
##    flops = tf.profiler.profile(graph=K.get_session().graph,
##                                run_meta=run_meta, cmd='op', options=opts)
##
##    return flops.total_float_ops  # Prints the "flops" of the model.
##
##pdb.set_trace()
##model.summary()
##print(get_flops(model))
#
#model.summary()

# Prepare callbacks for model saving and for learning rate adjustment.

# Saves the model after each epoch. Saved to the filepath, the latest best 
# model according to the quantity monitored will not be overwritten

global start_meas
start_meas = False

class RecordBatch(keras.callbacks.Callback):
    def __init__(self):
        super(RecordBatch, self).__init__()
        self.batch_time = {}
        self.batch_begin = 0
        self.curr_epoch = 0
        self.batch_num = args.start_batch + 1
        self.progress_time = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.curr_epoch = epoch
        self.batch_time[epoch] = []
    def on_epoch_end(self, epoch, logs=None):
        if int(time.time() - self.progress_time) >= 10:            
            progress = round((self.batch_num) / args.iters, 2)
            message = f'job {args.job_id} completion {progress}'
            send_signal(args.node, 10002, message)
            self.progress_time = time.time()
    def on_train_batch_begin(self, batch, logs=None):
        if self.batch_num == args.iters:
            message = f'job {args.job_id} finish'
            send_signal(args.node, 10002, message)
            time.sleep(0.5)
            sys.exit()
        self.batch_begin = time.time()
    def on_train_batch_end(self, batch, logs=None):
        duration = round(time.time() - self.batch_begin, 4)
        self.batch_num += 1
        if self.curr_epoch == starting_epoch and batch == 0:
            message = f'recover job {args.job_id}'
            send_signal(args.node, 10002, message)
            if args.mps_sync:
                print(f'{args.job_id} sent ready signal')
                with grpc.insecure_channel(f'localhost:{args.port}', options=(('grpc.enable_http_proxy', 0),)) as channel:
                    stub = grpc_pb2_grpc.SchedulerStub(channel)
                    response = stub.Notify(grpc_pb2.JobMessage(model=args.job_id, batch='', mode=''))

        global start_meas
        if start_meas:
            self.batch_time[self.curr_epoch].append(duration)

checkpoint_callback = CustomCheckpoint(
                        filepath=f'{save_file}/job{args.job_id}_epoch0.h5',
                        save_weights_only=True)

my_callback = RecordBatch()
if args.no_ckpt:
    callbacks = [my_callback]
else:
    callbacks = [my_callback, checkpoint_callback]

################### connects interrupt signal to the process #####################

def terminateProcess(signalNumber, frame):
    # first record the wasted epoch time
#    with open(f'logs/{gpu_type}_{args.job_id}{batch_size}.json', 'w') as f:
#        json.dump(my_callback.batch_time, f, indent=4)
    sys.exit()
def startProcess(signalNumber, frame):
    global start_meas
    start_meas = True

def ckptProcess(signalNumber, frame):
    curr_batch = my_callback.batch_num
    message = f'ckpt job {args.job_id} batch {curr_batch}' 
    send_signal(args.node, 10002, message)
    time.sleep(0.2)
    sys.exit()

signal.signal(signal.SIGINT, terminateProcess)
signal.signal(signal.SIGCONT, startProcess)
signal.signal(signal.SIGTERM, ckptProcess)

#if args.resume and starting_epoch > 0:
#    model.built = True
#    model.load_weights(saved)

#################################################################################

#model.save('/dev/shm/tmp/model.h5')
#model.save('/scratch/li.baol/MISO_Workload/resnet.h5')


#with open('/scratch/li.baol/MISO_Workload/data/cifar.npy', 'wb') as f:
#    np.save(f, x_train)
#    np.save(f, y_train)
#    np.save(f, x_test)
#    np.save(f, y_test)

# TODO load model instead of building

with open('/dev/shm/tmp/MISO_Workload/data/cifar.npy', 'rb') as f:
    x_train = np.load(f)
    y_train = np.load(f)
    x_test = np.load(f)
    y_test = np.load(f)

ts = time.time()
model = keras.models.load_model('/dev/shm/tmp/MISO_Workload/resnet.h5')
final = time.time() - ts

# Run training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=500000,
          validation_data=(x_test, y_test),
          shuffle=True,
        callbacks=callbacks,
        verbose=0,
        workers=4,
        initial_epoch=starting_epoch
)


# Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
