from datasets import load_dataset
from tensorflow import keras
import time
import json
import pandas as pd
import glob
import pdb
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers,losses,models,regularizers
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
import math

# two layer convolution
def conv_blk(input_tensor, num_filters, activation='relu', conv_kernel=(2,2), dropout=False):
    encoder = layers.Conv2D(num_filters, conv_kernel, padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    if dropout:
        encoder = layers.Dropout(0.1)(encoder)
    encoder = layers.Conv2D(num_filters, conv_kernel, padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activation)(encoder)
    if dropout:
        encoder = layers.Dropout(0.1)(encoder)
    return encoder

def encoder_blk(input_tensor, num_filters, activation='relu', conv_kernel=(2,2), pool_kernel=(2,2), pool_stride=(2,2), dropout=False):
    encoder = conv_blk(input_tensor, num_filters, activation, conv_kernel, dropout)
    encoder_pool = layers.MaxPooling2D(pool_kernel, strides=pool_stride)(encoder) # input too small, cannot 
    return encoder_pool, encoder

def decoder_blk(input_tensor, concat_tensor, num_filters, activation='relu', conv_kernel=(2,2), trans_kernel=(2,2), trans_stride=(2,2), dropout=False):
    decoder = layers.Conv2DTranspose(num_filters, trans_kernel, strides=trans_stride, padding='same')(input_tensor)
    if concat_tensor is not None:
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    decoder = layers.Conv2D(num_filters, conv_kernel, padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    if dropout:
        decoder = layers.Dropout(0.1)(decoder)
    decoder = layers.Conv2D(num_filters, conv_kernel, padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activation)(decoder)
    if dropout:
        decoder = layers.Dropout(0.1)(decoder)
    return decoder

def padding_model(input_shape=(3,7,1),
                 start_neurons=8, 
                 num_outputs=1, 
                 activation='relu',
                 dropout=False):
    inputs = layers.Input(shape=input_shape)

    # zero pad the input
    inputs_pad = layers.ZeroPadding2D(padding=((0,1),(0,1)))(inputs) # (4,8,1)

    encoder0_pool, encoder0 = encoder_blk(inputs_pad, start_neurons, activation=activation, dropout=dropout) # (2,4,8)
    encoder1_pool, encoder1 = encoder_blk(encoder0_pool, start_neurons*2, activation=activation, dropout=dropout) # (1,2,16)
    center = conv_blk(encoder1_pool, start_neurons*8, activation=activation) # (1,2,64)
    decoder1 = decoder_blk(center, encoder1, start_neurons*2, activation=activation, dropout=dropout) # (2,4,16)
    decoder0 = decoder_blk(decoder1, encoder0, start_neurons, activation=activation, dropout=dropout) # (4,8,8)
    outputs = layers.Conv2D(num_outputs, (1, 1), padding='same',
                                activation='linear', name='output_layer')(decoder0) # (4,8,1)

    # remove the padded data
    outputs = layers.Cropping2D(cropping=((0,1),(0,1)))(outputs)
        
    return tf.keras.models.Model(inputs=inputs,outputs=outputs,name='padding_model')


