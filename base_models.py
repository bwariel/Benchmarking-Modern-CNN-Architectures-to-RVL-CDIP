import os
import tensorflow as tf

from utils import CyclicLR, WarmUpCosineDecayScheduler, cosine_decay_with_warmup, step_decay_schedule, LRFinder, WarmUpLearningRateScheduler, SWA, OneCycleLR
import tensorflow as tf
import argparse
from efficientnet.keras import *

import csv
import cv2
import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, Input, Dense, GlobalAveragePooling2D, Flatten, Dropout, Conv2D, Add, BatchNormalization, Activation, AveragePooling2D, Cropping2D
#from keras.layers.merge import concatenate
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import *
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image
from keras.models import Model

from keras.callbacks import *
from keras.utils import multi_gpu_model
from keras.utils import to_categorical
from keras.utils import plot_model


import math
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import random
import subprocess
from subprocess import check_output
from scipy import ndimage
from skimage.io import imread
import sklearn.metrics as metrics
import tarfile
import time
from time import sleep
from tqdm import tqdm

from keras.preprocessing.image import save_img

from keras.activations import *

#os.chdir('./img_output')


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pickle
import time, os, fnmatch, shutil
from random_eraser import get_random_eraser


def get_architecture(model_name="EfficientNetB0", input_dim_width=224, input_dim_length=224,num_dense_layers=0,num_dense_nodes=0,num_class=16,dropout_pct=0.2, weights=None):

    # priors to use for base architecture
    def create_normal_residual_block(inputs, ch, N):
        # Conv with skip connections
        x = inputs
        for i in range(N):
            # adjust channels
            if i == 0:
                skip = Conv2D(ch, 1)(x)
                skip = BatchNormalization()(skip)
                skip = Activation("relu")(skip)
            else:
                skip = x
            x = Conv2D(ch, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(ch, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Add()([x, skip])
        return x



    def wide_resnet(N=1, k=1,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class):
        """
        Create vanilla conv Wide ResNet (N=4, k=10)
        """
        # input
        input =Input((input_dim_width,input_dim_length,3))
        # 16 channels block
        x = Conv2D(16, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # 1st block
        x = create_normal_residual_block(x, 16*k, N)
        # The original wide resnet is stride=2 conv for downsampling,
        # but replace them to average pooling because centers are shifted when octconv
        # 2nd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 32*k, N)
        # 3rd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 64*k, N)
        # FC
        x = GlobalAveragePooling2D()(x)
        x = Dense(num_class, activation="softmax")(x)

        model = Model(input, x)
        return model



    def very_wide_resnet(N=1, k=1,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class):
        """
        Create vanilla conv Wide ResNet (N=4, k=10)
        """
        # input
        input =Input((input_dim_width,input_dim_length,3))
        # 16 channels block
        x = Conv2D(64, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # 1st block
        x = create_normal_residual_block(x, 64*k, N)
        # The original wide resnet is stride=2 conv for downsampling,
        # but replace them to average pooling because centers are shifted when octconv
        # 2nd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 128*k, N)
        # 3rd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 256*k, N)
        # FC
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_pct)(x)
        x = Dense(num_class, activation="softmax")(x)

        model = Model(input, x)
        return model




    if model_name == "WRN":
        print("WRN")
        #model = wide_resnet(N=1, k=1,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class)
        input =Input((input_dim_width,input_dim_length,3))
        N=1
        k=1
        # 16 channels block
        x = Conv2D(16, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # 1st block
        x = create_normal_residual_block(x, 16*k, N)
        # The original wide resnet is stride=2 conv for downsampling,
        # but replace them to average pooling because centers are shifted when octconv
        # 2nd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 32*k, N)
        # 3rd block
        x = AveragePooling2D(2)(x)
        x = create_normal_residual_block(x, 64*k, N)
        # FC
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_pct)(x)
        x = Dense(num_class, activation="softmax")(x)
        
        model= Model(input, x)
        model.summary()
        base_model = model.layers[-4]


    if model_name == "VWRN":
        model = very_wide_resnet(N=1, k=2,input_dim_width=input_dim_width, input_dim_length=input_dim_length,num_class=num_class)


    if model_name == "DIN_D1":

        input_shape = Input(shape=(input_dim_width, input_dim_length, 3))

        #holistic head
        tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(input_shape)
        tower_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_1)
        tower_1 = Conv2D(128, (3,3), padding='same', activation='relu')(tower_1)
        tower_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_1)
        tower_1 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
        tower_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_1)
        tower_1 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_1)
        tower_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_1)
        tower_1 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_1)
        tower_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_1)

        #header head
        tower_2 = Cropping2D(cropping=((int(0.2*input_dim_length//1),0), (0,0)))(input_shape) #trim 60 pixels off bottom
        tower_2 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_2)
        tower_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_2)
        tower_2 = Conv2D(128, (3,3), padding='same', activation='relu')(tower_2)
        tower_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_2)
        tower_2 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_2)
        tower_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_2)
        tower_2 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_2)
        tower_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_2)
        tower_2 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_2)
        tower_2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_2)

        #footer head
        tower_3 = Cropping2D(cropping=((0,int(0.2*input_dim_length//1)), (0,0)))(input_shape) #trim 60 pixels off top
        tower_3 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
        tower_3 = Conv2D(128, (3,3), padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
        tower_3 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
        tower_3 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
        tower_3 = Conv2D(512, (3,3), padding='same', activation='relu')(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)


        merged = concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = GlobalAveragePooling2D()(merged)
        x = Dropout(dropout_pct)(merged)
        out = Dense(num_class, activation='softmax')(x)
        model = Model(input_shape, out)


    if model_name == "vgg11":
        #from original paper 
        input_shape = Input(shape=(input_dim_width, input_dim_width, 3))
        x = Conv2D(64, (3,3), padding='same', activation='relu')(input_shape)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        #classifier head, modified for GAP 
        x= GlobalAveragePooling2D()(x)
        x = Dropout(dropout_pct)(x)
        out = Dense(num_class, activation='softmax')(x)
        base_model = Model(input_shape, out)


    if model_name == "very_shallow_net":
        #from original paper
        input_shape = Input(shape=(input_dim_width, input_dim_width, 3))
        x = Conv2D(64, (3,3), padding='same', activation='relu')(input_shape)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(x)
        x = Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2), strides=(1, 1), padding='same')(x)
        #classifier head, modified for GAP
        x= GlobalAveragePooling2D()(x)
        x = Dropout(dropout_pct)(x)
        out = Dense(num_class, activation='softmax')(x)
        base_model = Model(input_shape, out)



    elif model_name == "vgg16":
        base_model = VGG16(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "SEResNet18":
        base_model = SEResNet18(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "SEResNet34":
        base_model = SEResNet34(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "SEResNet50":
        base_model = SEResNet50(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "SEResNet101":
        base_model = SEResNet101(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "SEResNet152":
        base_model = SEResNet152(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "ResNet18":
        base_model = ResNet18(input_shape= (input_dim_width, input_dim_length, 3), weights=weights, include_top=False)
    elif model_name == "ResNet34":
        base_model = ResNet34(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif model_name == "ResNet50":
        base_model = ResNet50(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif model_name == "ResNet101":
        base_model = ResNet101(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif model_name == "ResNet152":
        base_model = ResNet152(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "DenseNet121":
        base_model = DenseNet121(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)

    elif str(model_name) == "DenseNet169":
        base_model = DenseNet169(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "DenseNet201":
        base_model = DenseNet201(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "SENet154":
        base_model = SENet154(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)

        base_model = ResNet50(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif model_name == "ResNet101":
        base_model = ResNet101(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif model_name == "ResNet152":
        base_model = ResNet152(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "DenseNet121":
        base_model = DenseNet121(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)

    elif str(model_name) == "DenseNet169":
        base_model = DenseNet169(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "DenseNet201":
        base_model = DenseNet201(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "SENet154":
        base_model = SENet154(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)

    elif str(model_name) == "ResNeXt50":
        base_model = ResNeXt50(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "SEResNeXt50":
        base_model = SEResNeXt50(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "ResNeXt101":
        base_model = SEResNeXt101(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)


    elif str(model_name) == "ResNeXt101":
        base_model = ResNeXt101(input_shape= (input_dim_length, input_dim_length, 3), weights=args.weights, include_top=False)
    elif str(model_name) == "InceptionResNetV2":
        base_model = InceptionResNetV2(input_shape= (input_dim_width, input_dim_length, 3), weights=args.weights, include_top=False)

    elif str(model_name) == "NASNetLarge":
        base_model = NASNetLarge(input_shape= (input_dim_length, input_dim_width, 3), weights=args.weights, include_top=False)

    elif str(model_name) == "EfficientNetB0":
        base_model = EfficientNetB0(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB0":
        base_model = EfficientNetB0(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB1":
        base_model = EfficientNetB1(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB2":
        base_model = EfficientNetB2(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB3":
        base_model = EfficientNetB3(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB4":
        base_model = EfficientNetB4(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB5":
        base_model = EfficientNetB5(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB6":
        base_model = EfficientNetB6(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    elif str(model_name) == "EfficientNetB7":
        base_model = EfficientNetB7(input_shape= (input_dim_length, input_dim_width, 3), weights=weights, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = keras.layers.ELU(alpha=1.0)(x)

    if num_dense_layers == 2:
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)

    elif num_dense_layers == 1:
        x = Dense(num_dense_nodes)(x)
        x = Dropout(dropout_pct)(x)

    elif dropout_pct > 0:
        x = Dropout(dropout_pct)(x)
        #x = keras.layers.AlphaDropout(dropout_pct, noise_shape=None, seed=None)(x)

    predictions = Dense(num_class, activation='softmax',name='predictions',kernel_initializer='glorot_normal', 
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
            kernel_constraint=None, bias_constraint=None)(x)
    model = Model(inputs=base_model.input, outputs=predictions, name = str(model_name))

    return model
