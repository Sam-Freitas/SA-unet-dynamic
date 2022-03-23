import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import shutil

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, data_generator_for_testing,load_first_image_get_size,get_num_layers_unet,dynamic_wnet_cnn
from SA_dynamic_unet import bwareafilt, bwareaopen

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "test")
# image_path = os.path.join('testing')
channels = 3
batch_size = 256
starting_filter_size = 16

img_size = load_first_image_get_size(image_path,force_img_size=128)
num_layers_of_unet, img_size = get_num_layers_unet(img_size)
height = width = img_size

num_layers_of_unet = num_layers_of_unet + 1

checkpoint_path = "model_checkpoints/cp.ckpt" 
print('Loading in model from best checkpoint')

new_model = dynamic_unet_cnn(height,width,channels,num_layers = num_layers_of_unet,starting_filter_size=starting_filter_size)
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.1)
new_model.compile(optimizer=optimizer_adam, loss='BinaryCrossentropy', metrics=['accuracy','MeanAbsoluteError'], run_eagerly = True)
new_model.load_weights(checkpoint_path)

# new_model.save('compiled_model/model_128img_16filt_final')
# del new_model
# new_model = tf.keras.models.load_model('compiled_model/model_64_final')

images_resized, images = data_generator_for_testing(image_path,height,width,channels,recursive = True,spcific_file_ext = 'jpg', normalize = True) #get test set
# images = images / 255 #thresh y_test

for count, image in enumerate(images_resized): #for loop for plotting images
    
    # resize image
    img = image.reshape((1,height,width,channels))
    # convert to proper data type
    img = img.astype(np.float64)
    # predict
    pred_mask = new_model.predict(img)
    # islate mask
    out_img = pred_mask[:,:,:,-1]
    # reshape mask to inital image size
    og_image_size = images[count].shape[:2]
    out_img_large = cv2.resize(out_img.squeeze(),og_image_size)
    # export
    plot_figures(images[count],out_img_large, count, BGR = True)
    count += 1

    plt.close('all')