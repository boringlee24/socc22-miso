# TensorFlow and tf.keras
import tensorflow as tf
import os
import argparse
from tensorflow import keras
import time
import json 
from pathlib import Path
import signal
import subprocess
from datetime import datetime
import pdb
import sys
import grpc
home = os.environ.get('HOME')
sys.path.append(f'{home}/GIT/socc22-miso/mps/grpc')
import grpc_pb2, grpc_pb2_grpc

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('--gpu_type', metavar='GPU', type=str, help='specific model name')
parser.add_argument('--partition', metavar='par', type=str, help='thread partition percentage', default='100')
parser.add_argument('--time_limit', metavar='time', type=int, help='run for limited time', default=310)
parser.add_argument('--port', metavar='pt', type=int, help='port for grpc', default=50051)
parser.add_argument('--direct_start', dest='direct_start', action='store_true')
parser.set_defaults(direct_start=False)
args = parser.parse_args()

os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition 
gpu_type = args.gpu_type
model_name = sys.argv[0].split('.')[0]
batch_size = 32

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5000, callbacks=callbacks,
        verbose=0, 
        workers=4)

