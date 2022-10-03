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
from keras.datasets import cifar10
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import models, layers, optimizers
from datetime import datetime
import tensorflow as tf
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
sys.path.append(f'{home}/GIT/socc22-miso/mps/grpc')
import grpc_pb2, grpc_pb2_grpc

cwd = os.getcwd()
user = cwd.split('GIT')[0]
model_name = sys.argv[0].split('.')[0]

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', metavar='LEARNING_RATE', type=float, help='learning rate', default=0.001)
parser.add_argument('--gpu_type', metavar='GPU', type=str, help='specific model name')
parser.add_argument('--partition', metavar='par', type=str, help='thread partition percentage', default='100')
parser.add_argument('--time_limit', metavar='time', type=int, help='run for limited time', default=310)
parser.add_argument('--port', metavar='pt', type=int, help='port for grpc', default=50051)
parser.add_argument('--direct_start', dest='direct_start', action='store_true')
parser.set_defaults(direct_start=False)

args = parser.parse_args()

os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition 
gpu_type = args.gpu_type
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
# Training parameters
batch_size = args.batch_size  # orig paper trained all networks with batch_size=128
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

n = 3

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


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

model = models.Sequential()

base_model = MobileNetV2(weights=None, include_top=False, input_shape=(32, 32, 3), pooling=None)

#base_model.summary()

#pdb.set_trace()

#model.add(layers.UpSampling2D((2,2)))
#model.add(layers.UpSampling2D((2,2)))
#model.add(layers.UpSampling2D((2,2)))
model.add(base_model)
model.add(layers.Flatten())
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))#, kernel_initializer='he_uniform'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=1, batch_size=20, validation_data=(x_test, y_test))

#def get_flops(model):
#    run_meta = tf.RunMetadata()
#    opts = tf.profiler.ProfileOptionBuilder.float_operation()
#
#    # We use the Keras session graph in the call to the profiler.
#    flops = tf.profiler.profile(graph=K.get_session().graph,
#                                run_meta=run_meta, cmd='op', options=opts)
#
#    return flops.total_float_ops  # Prints the "flops" of the model.
#
#pdb.set_trace()
#model.summary()
#print(get_flops(model))

model.summary()

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
        self.epoch_time = {}
        self.start_time = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.curr_epoch = epoch
        self.batch_time[epoch] = []
        now = datetime.now()
        self.epoch_time[epoch] = str(now)
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_begin = time.time()
    def on_train_batch_end(self, batch, logs=None):
        duration = round(time.time() - self.batch_begin, 4)
        global start_meas
        if not start_meas:
            if not args.direct_start:
                print(f'{model_name} waiting for start signal')
                # grpc
                with grpc.insecure_channel(f'localhost:{args.port}', options=(('grpc.enable_http_proxy', 0),)) as channel:
                    stub = grpc_pb2_grpc.SchedulerStub(channel)
                    response = stub.Notify(grpc_pb2.JobMessage(model=model_name, batch=str(batch_size), mode=gpu_type))
            else:
                start_meas = True
            while not start_meas:
                time.sleep(0.001)
            self.start_time = time.time()
        self.batch_time[self.curr_epoch].append(duration)

        if time.time() - self.start_time >= args.time_limit:
            os.kill(os.getpid(), signal.SIGINT)
    def on_train_end(self, logs=None):
        # write to json
        with open(f'logs/{gpu_type}_{model_name}{batch_size}.json', 'w') as f:
            json.dump(self.batch_time, f, indent=4)

my_callback = RecordBatch()
callbacks = [my_callback]

################### connects interrupt signal to the process #####################

def terminateProcess(signalNumber, frame):
    # first record the wasted epoch time
    with open(f'logs/{gpu_type}_{model_name}{batch_size}.json', 'w') as f:
        json.dump(my_callback.batch_time, f, indent=4)
    sys.exit()
def startProcess(signalNumber, frame):
    global start_meas
    start_meas = True

signal.signal(signal.SIGINT, terminateProcess)
signal.signal(signal.SIGCONT, startProcess)

#################################################################################

# Run training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=500,
          validation_data=(x_test, y_test),
          shuffle=True,
        callbacks=callbacks,
        verbose=0, 
        workers=4)


# Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
