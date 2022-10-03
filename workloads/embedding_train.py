import socket
import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import json
import signal
import sys
import subprocess
import argparse
import os
import pathlib
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
    # pdb.set_trace()
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

# data_path = keras.utils.get_file(
#    "news20.tar.gz",
#    "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
#    untar=True,
# )

# data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
# dirnames = os.listdir(data_dir)
# print("Number of directories:", len(dirnames))
# print("Directory names:", dirnames)

# fnames = os.listdir(data_dir / "comp.graphics")
# print("Number of files in comp.graphics:", len(fnames))
# print("Some example filenames:", fnames[:5])
# samples = []
# labels = []
# class_names = []
# class_index = 0
# for dirname in sorted(os.listdir(data_dir)):
#    class_names.append(dirname)
#    dirpath = data_dir / dirname
#    fnames = os.listdir(dirpath)
#    print("Processing %s, %d files found" % (dirname, len(fnames)))
#    for fname in fnames:
#        fpath = dirpath / fname
#        f = open(fpath, encoding="latin-1")
#        content = f.read()
#        lines = content.split("\n")
#        lines = lines[10:]
#        content = "\n".join(lines)
#        samples.append(content)
#        labels.append(class_index)
#    class_index += 1

# print("Classes:", class_names)
# print("Number of samples:", len(samples))

# """
# ## Shuffle and split the data into training & validation sets
# """

# # Shuffle the data
# seed = 1337
# rng = np.random.RandomState(seed)
# rng.shuffle(samples)
# rng = np.random.RandomState(seed)
# rng.shuffle(labels)

# # Extract a training & validation split
# validation_split = 0.2
# num_validation_samples = int(validation_split * len(samples))
# train_samples = samples[:-num_validation_samples]
# val_samples = samples[-num_validation_samples:]
# train_labels = labels[:-num_validation_samples]
# val_labels = labels[-num_validation_samples:]

# """
# ## Create a vocabulary index

# Let's use the `TextVectorization` to index the vocabulary found in the dataset.
# Later, we'll use the same layer instance to vectorize the samples.

# Our layer will only consider the top 20,000 words, and will truncate or pad sequences to
# be actually 200 tokens long.
# """

# from tensorflow.keras.layers import TextVectorization

# vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
# text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(batch_size)
# vectorizer.adapt(text_ds)

# """
# You can retrieve the computed vocabulary used via `vectorizer.get_vocabulary()`. Let's
# print the top 5 words:
# """

# vectorizer.get_vocabulary()[:5]

# """
# Let's vectorize a test sentence:
# """

# output = vectorizer([["the cat sat on the mat"]])
# output.numpy()[0, :6]

# """
# As you can see, "the" gets represented as "2". Why not 0, given that "the" was the first
# word in the vocabulary? That's because index 0 is reserved for padding and index 1 is
# reserved for "out of vocabulary" tokens.

# Here's a dict mapping words to their indices:
# """

# voc = vectorizer.get_vocabulary()
# word_index = dict(zip(voc, range(len(voc))))

# """
# As you can see, we obtain the same encoding as above for our test sentence:
# """

# test = ["the", "cat", "sat", "on", "the", "mat"]
# [word_index[w] for w in test]

# """
# ## Load pre-trained word embeddings
# """

# """
# Let's download pre-trained GloVe embeddings (a 822M zip file).

# You'll need to run the following commands:

# ```
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip -q glove.6B.zip
# ```
# """

# """
# The archive contains text-encoded vectors of various sizes: 50-dimensional,
# 100-dimensional, 200-dimensional, 300-dimensional. We'll use the 100D ones.

# Let's make a dict mapping words (strings) to their NumPy vector representation:
# """

# path_to_glove_file = os.path.join(
#    os.path.expanduser(f"/work/{user}"), "keras/datasets/glove.6B.100d.txt"
# )

# embeddings_index = {}
# with open(path_to_glove_file) as f:
#    for line in f:
#        word, coefs = line.split(maxsplit=1)
#        coefs = np.fromstring(coefs, "f", sep=" ")
#        embeddings_index[word] = coefs

# print("Found %s word vectors." % len(embeddings_index))

# """
# Now, let's prepare a corresponding embedding matrix that we can use in a Keras
# `Embedding` layer. It's a simple NumPy matrix where entry at index `i` is the pre-trained
# vector for the word of index `i` in our `vectorizer`'s vocabulary.
# """

# num_tokens = len(voc) + 2
# embedding_dim = 100
# hits = 0
# misses = 0

# # Prepare embedding matrix
# embedding_matrix = np.zeros((num_tokens, embedding_dim))
# for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # Words not found in embedding index will be all-zeros.
#        # This includes the representation for "padding" and "OOV"
#        embedding_matrix[i] = embedding_vector
#        hits += 1
#    else:
#        misses += 1
# print("Converted %d words (%d misses)" % (hits, misses))


# """
# Next, we load the pre-trained word embeddings matrix into an `Embedding` layer.

# Note that we set `trainable=False` so as to keep the embeddings fixed (we don't want to
# update them during training).
# """

# from tensorflow.keras.layers import Embedding

# embedding_layer = Embedding(
#    num_tokens,
#    embedding_dim,
#    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#    trainable=False,
# )

# """
# ## Build the model

# A simple 1D convnet with global max pooling and a classifier at the end.
# """

# from tensorflow.keras import layers

# int_sequences_input = keras.Input(shape=(None,), dtype="int64")
# embedded_sequences = embedding_layer(int_sequences_input)
# x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(128, 5, activation="relu")(x)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(128, 5, activation="relu")(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dense(128, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
# preds = layers.Dense(len(class_names), activation="softmax")(x)
# model = keras.Model(int_sequences_input, preds)
# model.summary()

# """
# ## Train the model

# First, convert our list-of-strings data to NumPy arrays of integer indices. The arrays
# are right-padded.
# """

# x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
# x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

# y_train = np.array(train_labels)
# y_val = np.array(val_labels)

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size).take(120)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).take(120)

# train_dataset = train_dataset.concatenate(train_dataset).concatenate(train_dataset).concatenate(train_dataset)


# """
# We use categorical crossentropy as our loss since we're doing softmax classification.
# Moreover, we use `sparse_categorical_crossentropy` since our labels are integers.
# """

# model.compile(
#    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
# )

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

# tf.data.experimental.save(train_dataset, f'/work/li.baol/MISO_Workload/news20_train/b{batch_size}')
# tf.data.experimental.save(test_dataset, f'/work/li.baol/MISO_Workload/news20_eval/b{batch_size}')
# pdb.set_trace()
train_dataset = tf.data.experimental.load(f'/dev/shm/tmp/MISO_Workload/news20_train/b{batch_size}')
test_dataset = tf.data.experimental.load(f'/dev/shm/tmp/MISO_Workload/news20_eval/b{batch_size}')
model = keras.models.load_model('/dev/shm/tmp/MISO_Workload/embedding.h5')

model.fit(train_dataset,
        epochs=500000,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=0, 
        workers=4,
        initial_epoch=starting_epoch
)
#model.save('/scratch/li.baol/keras/embedding.h5')


