"""
Title: Timeseries classification with a Transformer model
Author: [Theodoros Ntakouris](https://github.com/ntakouris)
Date created: 2021/06/25
Last modified: 2021/08/05
Description: This notebook demonstrates how to do timeseries classification using a Transformer model.
"""


"""
## Introduction

This is the Transformer architecture from
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
applied to timeseries instead of natural language.

This example requires TensorFlow 2.4 or higher.

## Load the dataset

We are going to use the same dataset and preprocessing as the
[TimeSeries Classification from Scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch)
example.
"""

import socket
import numpy as np
import pdb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import argparse
import time
import json 
from pathlib import Path
import signal
import subprocess
from datetime import datetime
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
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--job_id', metavar='JOB', type=str, help='ID of job from trace', default='0')
parser.add_argument('--partition', metavar='par', type=str, help='thread partition percentage', default='100')
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

batch_size = args.batch_size
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# def readucr(filename):
#    data = np.loadtxt(filename, delimiter="\t")
#    y = data[:, 0]
#    x = data[:, 1:]
#    return x, y.astype(int)

# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
# x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = 2 #len(np.unique(y_train))
#pdb.set_trace()
#
# idx = np.random.permutation(len(x_train))
# x_train = x_train[idx]
# y_train = y_train[idx]

# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size).take(25)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).take(25)

"""
## Build the model

Our model processes a tensor of shape `(batch size, sequence length, features)`,
where `sequence length` is the number of time steps and `features` is each input
timeseries.

You can replace your classification RNN layers with this one: the
inputs are fully compatible!
"""


"""
We include residual connections, layer normalization, and dropout.
The resulting layer can be stacked multiple times.

The projection layers are implemented through `keras.layers.Conv1D`.
"""

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

"""
The main part of our model is now complete. We can stack multiple of those
`transformer_encoder` blocks and we can also proceed to add the final
Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
layers, we need to reduce the output tensor of the `TransformerEncoder` part of
our model down to a vector of features for each data point in the current
batch. A common way to achieve this is to use a pooling layer. For
this example, a `GlobalAveragePooling1D` layer is sufficient.
"""


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


"""
## Train and evaluate
"""

input_shape = (500,1)#x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
#model.summary()

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

##################################################################################

# tf.data.experimental.save(train_dataset, f'/work/li.baol/MISO_Workload/FordA_train/b{batch_size}/')
# tf.data.experimental.save(test_dataset, f'/work/li.baol/MISO_Workload/FordA_eval/b{batch_size}/')

train_dataset = tf.data.experimental.load(f'/dev/shm/tmp/MISO_Workload/FordA_train/b{batch_size}/')
test_dataset = tf.data.experimental.load(f'/dev/shm/tmp/MISO_Workload/FordA_eval/b{batch_size}/')

model.fit(train_dataset,
        epochs=500000,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=0, 
        workers=4,
        initial_epoch=starting_epoch
)
#model.save('/scratch/li.baol/keras/transformer.h5')

