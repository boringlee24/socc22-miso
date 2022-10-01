from datasets import load_dataset
from tensorflow import keras
import time
import json
import signal
import sys
import subprocess
import argparse
import os
import grpc
home = os.environ.get('HOME')
sys.path.append(f'{home}/GIT/mig_exp/mps/grpc')
import grpc_pb2, grpc_pb2_grpc

#os.environ['CUDA_MPS_PIPE_DIRECTORY'] = f'/scratch/{os.environ['USER']}/nvidia-mps'
#os.environ['CUDA_MPS_LOG_DIRECTORY'] = f'/scratch/{os.environ['USER']}/nvidia-log'

model_name = sys.argv[0].split('.')[0]

parser = argparse.ArgumentParser(description='BERT')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--gpu_type', metavar='GPU', type=str, help='specific model name')
parser.add_argument('--partition', metavar='par', type=str, help='thread partition percentage', default='100')
parser.add_argument('--time_limit', metavar='time', type=int, help='run for limited time', default=310)
parser.add_argument('--port', metavar='pt', type=int, help='port for grpc', default=50051)
parser.add_argument('--direct_start', dest='direct_start', action='store_true')
parser.set_defaults(direct_start=False)
args = parser.parse_args()

os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = args.partition 

gpu_type = args.gpu_type
batch_size = args.batch_size
raw_datasets = load_dataset("imdb")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#inputs = tokenizer(sentences, padding="max_length", truncation=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

tf_train_dataset = small_train_dataset.remove_columns(["text"]).with_format("tensorflow")
tf_eval_dataset = small_eval_dataset.remove_columns(["text"]).with_format("tensorflow")

train_features = {x: tf_train_dataset[x].to_tensor() for x in tokenizer.model_input_names}
train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, tf_train_dataset["label"]))
train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset)).batch(batch_size)

eval_features = {x: tf_eval_dataset[x].to_tensor() for x in tokenizer.model_input_names}
eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["label"]))
eval_tf_dataset = eval_tf_dataset.batch(batch_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

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

model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=200, callbacks=callbacks, verbose=0, workers=4)
