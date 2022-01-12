import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import shutil

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, data_generator_for_testing,load_first_image_get_size,get_num_layers_unet,dynamic_wnet_cnn
from SA_dynamic_unet import bwareafilt

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "test")
# image_path = os.path.join('testing')
channels = 3
batch_size = 1

img_size = load_first_image_get_size(image_path,force_img_size=128)
num_layers_of_unet, img_size = get_num_layers_unet(img_size)
height = width = img_size

starting_kernal_size = 16

checkpoint_path = "training_unet/cp.ckpt" 
print('Loading in model from best checkpoint')

new_model = dynamic_unet_cnn(height,width,channels,num_layers = num_layers_of_unet)
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.1)
new_model.compile(optimizer=optimizer_adam, loss='BinaryCrossentropy', metrics=['accuracy','MeanAbsoluteError'], run_eagerly = True)
new_model.load_weights(checkpoint_path)

images = data_generator_for_testing(image_path,height,width,channels,recursive = True,spcific_file_ext = 'jpg', normalize = True) #get test set
# images = images / 255 #thresh y_test

count = 1 #counter for figures in for loops
for image in images: #for loop for plotting images
    
    img = image.reshape((1,height,width,channels))
    img = img.astype(np.float64)
    pred_mask = new_model.predict(img)

    # bwfilt = bwareafilt((pred_mask[:,:,:,-1]>0.5)*1,n=5)

    # out_img = bwfilt[0]

    out_img = pred_mask[:,:,:,-1]

    plot_figures(image,out_img, count)
    count += 1

    plt.close('all')