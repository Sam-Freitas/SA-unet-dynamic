import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import os
import sys
from sklearn.utils import validation
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, plot_acc_loss, data_generator, load_first_image_get_size, get_num_layers_unet, test_on_improved_val_loss

plt.ion() #turn ploting on

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "images")
mask_path = os.path.join(dataset_path,"masks")
dataset = pd.read_csv('dataset.csv')

total = len(dataset) #set variables
test_split = 5/total
# height = 512
# width = 512
channels = 3
batch_size = 1

img_size = load_first_image_get_size(image_path,dataset,force_img_size=128)
num_layers_of_unet, img_size = get_num_layers_unet(img_size)
height = width = img_size

starting_kernal_size = 16

model = dynamic_unet_cnn(height,width,channels,
    num_layers = num_layers_of_unet,starting_filter_size = starting_kernal_size, use_dropout = True,
    num_classes=2)
optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer_adam, loss='CategoricalCrossentropy', metrics=['accuracy','MeanAbsoluteError'], run_eagerly = True)
# model.summary() #display model summary, Better to just view the model.png

try:
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
except:
    print("Exporting model to png failed")
    print("Necessary packages: pydot (pip) and graphviz (brew)")

#######Training
train, test = train_test_split(dataset, test_size = test_split, random_state = 50) #randomly split up the test and training datasets
X_train, y_train = data_generator(train, image_path, mask_path, height, width, channels) #set up training data
y_train = y_train / 255 #thresh y_training set
y_train_cat = tf.keras.utils.to_categorical(y_train)
X_train = X_train / 255

checkpoint_path = "training_1/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor="val_loss",mode="min",
    save_best_only = True,verbose=1,save_weights_only=True) #use checkpoint instead of sequential() module
earlystop = EarlyStopping(monitor = 'val_loss', 
    patience = 50, verbose = 1,restore_best_weights = True) #stop at best epoch
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
    patience=25, min_lr=0.0001, verbose = 1)

on_e_end = test_on_improved_val_loss()

results = model.fit(X_train, y_train_cat, validation_split=0.1, batch_size=batch_size, 
    epochs=250,callbacks=[earlystop, checkpoint,reduce_lr,on_e_end]) #fit model

# plot_acc_loss(results) #plot the accuracy and loss functions

del model
del X_train
del y_train

print('Loading in model from best checkpoint')
new_model = dynamic_unet_cnn(height,width,channels,
    num_layers = num_layers_of_unet,starting_filter_size = starting_kernal_size, use_dropout = False,num_classes=2)
new_model.compile(optimizer=optimizer_adam, loss='CategoricalCrossentropy', metrics=['accuracy','MeanAbsoluteError'], run_eagerly = True)
new_model.load_weights(checkpoint_path)

X_test,y_test = data_generator(test,image_path, mask_path,height,width,channels) #get test set
y_test = y_test / 255 #thresh y_test
y_test_cat = tf.keras.utils.to_categorical(y_test)
X_test = X_test / 255
eval_result = new_model.evaluate(X_test,y_test_cat,steps=1) #get evaluation results
print("Restored model, accuracy: {:5.2f}%".format(100 * eval_result[1]))

count = 1 #counter for figures in for loops
for image,mask in zip(X_test,y_test): #for loop for plotting images
    
    img = image.reshape((1,height,width,channels))
    pred_mask = new_model.predict(img)

    plot_figures(image,pred_mask[:,:,:,-1], count, orig_mask=mask, ext = 'training')
    plt.close('all')
    count += 1

plt.ioff()
plt.show()
