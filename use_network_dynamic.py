#Courtney Comrie and Sam Freitas
#ECE 523 Final Project
#UNET-CNN for segmenting brains from the skull

import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import os
import sys
from tqdm import tqdm
import random
import warnings
from itertools import chain
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import shutil
import glob
from natsort import natsorted

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, data_generator_for_testing

plt.ion() #turn ploting on

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "testing")
dataset = pd.read_csv('dataset.csv')

total = len(dataset) #set variables
test_split = 0.2
height = 128
width = 128
channels = 1 
batch_size = 32

## 128 - 2
## 512 - 4
## 1024 - 6 ???????

num_layers_of_unet = 2
starting_kernal_size = 16

checkpoint_path = "training_1/cp.ckpt" 
print('Loading in model from best checkpoint')
new_model = dynamic_unet_cnn(height,width,channels,
    num_layers = num_layers_of_unet,starting_filter_size = starting_kernal_size, use_dropout = False)
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
new_model.load_weights(checkpoint_path)

images = data_generator_for_testing(image_path,height,width,channels, num_to_load = 3) #get test set
images = images / 255 #thresh y_test

output_path = os.path.join(os.getcwd(),'output_images')
try:
    os.mkdir(output_path)
except:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

count = 1 #counter for figures in for loops
for image in images: #for loop for plotting images
    
    img = image.reshape((1,height,width,channels))
    img = img.astype(np.float64)
    pred_mask = new_model.predict(img)

    plot_figures(image,pred_mask, count, ext = 'testing')
    count += 1

    plt.close('all')

plt.ioff()
plt.show()