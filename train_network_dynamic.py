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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, plot_acc_loss, data_generator, bwareafilt
from SA_dynamic_unet import load_first_image_get_size, get_num_layers_unet, test_on_improved_val_loss,dynamic_wnet_cnn

plt.ion() #turn ploting on

random.seed(50)
np.random.seed(50)

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "images")
mask_path = os.path.join(dataset_path,"masks")
dataset = pd.read_csv('dataset.csv')

total = len(dataset) #set variables
test_split = 100/total
channels = 3
batch_size = 64
starting_filter_size = 4

img_size = load_first_image_get_size(image_path,dataset,force_img_size=128)
num_layers_of_unet, img_size = get_num_layers_unet(img_size)
height = width = img_size

#######Training
train, test = train_test_split(dataset, test_size = test_split, random_state = 50) #randomly split up the test and training datasets
X_train, y_train = data_generator(train, image_path, mask_path, height, width, channels, create_more_data=True, data_multiplication = 3, normalize = True) #set up training data
# y_train = y_train / 255 #thresh y_training set
# # y_train_cat = tf.keras.utils.to_categorical(y_train)
# X_train = X_train / 255

X_val = X_train[0:train.shape[0]]
y_val = y_train[0:train.shape[0]]

X_train = X_train[train.shape[0]::]
y_train = y_train[train.shape[0]::]

shuffled_idx = np.arange(X_train.shape[0])
np.random.shuffle(shuffled_idx)

X_train = X_train[shuffled_idx]
y_train = y_train[shuffled_idx]

output = pd.DataFrame()

model = dynamic_unet_cnn(height,width,channels,
        num_layers = num_layers_of_unet, starting_filter_size = starting_filter_size, use_dropout = True, num_classes=1)

def scheduler(epoch,lr):
    if epoch < 150:
        lr = 0.01
    elif epoch > 150 and epoch < 500:
        lr = 0.0005
    else:
        lr = 0.0001

    return lr 
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

optimizer_adam = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()#BinaryFocalCrossentropy()
AUC_ROC = tf.keras.metrics.AUC(curve='ROC',name='AUC_ROC')
AUC_PR = tf.keras.metrics.AUC(curve='PR',name='AUC_PR',num_thresholds = 25)
model.compile(optimizer=optimizer_adam, loss=loss, metrics=[AUC_PR], run_eagerly = True)

checkpoint_path = "model_testing/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)

epochs = 10000
checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor="val_AUC_PR",mode="max",
    save_best_only = True,verbose=1,save_weights_only=True) #use checkpoint instead of sequential() module
earlystop = EarlyStopping(monitor = 'val_AUC_PR', min_delta=0,
    patience = 400, verbose = 1,restore_best_weights = True,mode = 'max') #stop at best epoch
# reduce_lr = ReduceLROnPlateau(monitor='val_AUC_PR', mode = 'max',factor=0.9,min_delta=0.0,
#     patience=3, min_lr=0.005, cooldown = 5,verbose = 1)

on_e_end = test_on_improved_val_loss()

results = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=batch_size, 
    epochs=epochs,callbacks=[earlystop, on_e_end, checkpoint,lr_callback],verbose = 1) #fit model

# plot_acc_loss(results) #plot the accuracy and loss functions

del model

# print('Loading in model from best checkpoint')
new_model = dynamic_unet_cnn(height,width,channels,
    num_layers = num_layers_of_unet,starting_filter_size = starting_filter_size, use_dropout = False,num_classes=1)
new_model.compile(optimizer=optimizer_adam, loss=loss, metrics=['accuracy','MeanAbsoluteError',AUC_PR], run_eagerly = True)
new_model.load_weights(checkpoint_path)

X_test,y_test = data_generator(test, image_path, mask_path, height, width, channels, normalize = True,disable_bar=True)
eval_result = new_model.evaluate(X_test,y_test,steps=1,return_dict = True) #get evaluation results
eval_result['Filter_size'] = starting_filter_size

output = output.append(eval_result, ignore_index=True)
print(eval_result)

# output.to_csv('output.csv',index = False)

count = 1 #counter for figures in for loops
for image,mask in zip(X_test,y_test): #for loop for plotting images
    
    img = image.reshape((1,height,width,channels))
    pred_mask = new_model.predict(img)

    plot_figures(image,pred_mask[:,:,:,-1]>0.75, count, orig_mask=mask, ext = 'training', BGR = True)
    plt.close('all')
    count += 1

plt.ioff()
plt.show()
