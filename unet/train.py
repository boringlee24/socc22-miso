from tensorflow import keras
import time
import json
import pandas as pd
import glob
import pdb
import numpy as np
import tensorflow as tf
import os
from unet_utils import *
from pathlib import Path

unet = padding_model(start_neurons=32, activation='relu', dropout=False) 
params={
    'start_neurons'   :32,      # Controls size of hidden layers in CNN, higher = more complexity 
    'activation'      :'relu',  # Activation used throughout the U-Net,  see https://www.tensorflow.org/api_docs/python/tf/keras/activations
    'loss'            :'mae',   # Either 'mae' or 'mse', or others as https://www.tensorflow.org/api_docs/python/tf/keras/losses
    'loss_weights'    : 1,    # Scale for loss.  Recommend squaring this if using MSE
    'opt'             :tf.keras.optimizers.Adam,  # optimizer, see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    'learning_rate'   :0.05718,   # Learning rate for optimizer
    'num_epochs'      :50,       # Number of epochs to train for
    'batch'           :53
}
opt=params['opt'](learning_rate=params['learning_rate'], amsgrad=False)
unet.compile(optimizer=opt, loss=params['loss'], loss_weights=[params['loss_weights']], metrics=[tf.keras.metrics.MeanAbsoluteError()])

############# data ###############
X_data = np.load('input_data.npy')
X_data = np.expand_dims(X_data, axis=-1)
Y_data = np.load('target_data.npy')
Y_data = np.expand_dims(Y_data, axis=-1)

tensorslice = tf.data.Dataset.from_tensor_slices((X_data,Y_data)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=False).batch(params['batch'])
train_data = tensorslice.skip(int(len(tensorslice) * 0.25)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=True)
test_data = tensorslice.take(int(len(tensorslice) * 0.25)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=True)
##################################

training_dir = 'training_dir'
testcase = 'tuned_unet'
# Path(f'{training_dir}/tensorboard/{testcase}').mkdir(parents=True, exist_ok=True)

callbacks=[
#    tf.keras.callbacks.ModelCheckpoint(training_dir+f'/{testcase}.hdf5', 
#                    monitor='val_loss',save_best_only=True),
#    tf.keras.callbacks.TensorBoard(log_dir=f'{training_dir}/tensorboard/{testcase}'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.2)
]

history1 = unet.fit(train_data, validation_data=test_data,
                  epochs=params['num_epochs'],
                  callbacks=callbacks,
                  workers=8)

unet.save(f'{training_dir}/unet.h5')
