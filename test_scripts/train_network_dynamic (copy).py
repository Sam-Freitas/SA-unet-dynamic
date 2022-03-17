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

from SA_dynamic_unet import dynamic_unet_cnn, plot_figures, plot_acc_loss, data_generator 
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
batch_size = 16

img_size = load_first_image_get_size(image_path,dataset,force_img_size=128)
num_layers_of_unet, img_size = get_num_layers_unet(img_size)
height = width = img_size

#######Training
train, test = train_test_split(dataset, test_size = test_split, random_state = 50) #randomly split up the test and training datasets
X_train, y_train = data_generator(train, image_path, mask_path, height, width, channels, create_more_data=True, data_multiplication = 3, normalize = True) #set up training data
# y_train = y_train / 255 #thresh y_training set
# # y_train_cat = tf.keras.utils.to_categorical(y_train)
# X_train = X_train / 255

X_val = X_train[0:500]
y_val = y_train[0:500]

X_train = X_train[500::]
y_train = y_train[500::]

shuffled_idx = np.arange(X_train.shape[0])
np.random.shuffle(shuffled_idx)

X_train = X_train[shuffled_idx]
y_train = y_train[shuffled_idx]

output = pd.DataFrame()

for starting_kernal_size in [4,8,16,32,64,128]:

    for i in range(5):

        model = dynamic_unet_cnn(height,width,channels,
                num_layers = num_layers_of_unet, starting_filter_size = starting_kernal_size, use_dropout = True, num_classes=1)

        optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
        AUC_ROC = tf.keras.metrics.AUC(curve='ROC',name='AUC_ROC')
        AUC_PR = tf.keras.metrics.AUC(curve='PR',name='AUC_PR',num_thresholds = 25)
        model.compile(optimizer=optimizer_adam, loss='BinaryCrossentropy', metrics=[AUC_PR], run_eagerly = True)

        checkpoint_path = "training_unet/cp.ckpt" 
        checkpoint_dir = os.path.dirname(checkpoint_path)

        epochs = 5
        checkpoint = ModelCheckpoint(filepath = checkpoint_path,monitor="val_AUC_PR",mode="max",
            save_best_only = True,verbose=0,save_weights_only=True) #use checkpoint instead of sequential() module
        earlystop = EarlyStopping(monitor = 'val_AUC_PR', min_delta=0,
            patience = 250, verbose = 1,restore_best_weights = True) #stop at best epoch
        reduce_lr = ReduceLROnPlateau(monitor='val_AUC_PR', factor=0.9,min_delta=0,
            patience=2, min_lr=0.00001, verbose = 0)

        on_e_end = test_on_improved_val_loss()

        print("Fitting model",i)
        results = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=batch_size, 
            epochs=epochs,callbacks=[earlystop, reduce_lr, checkpoint],verbose = 0) #fit model

        # plot_acc_loss(results) #plot the accuracy and loss functions

        del model

        # print('Loading in model from best checkpoint')
        new_model = dynamic_unet_cnn(height,width,channels,
            num_layers = num_layers_of_unet,starting_filter_size = starting_kernal_size, use_dropout = False,num_classes=1)
        new_model.compile(optimizer=optimizer_adam, loss='BinaryCrossentropy', metrics=['accuracy','MeanAbsoluteError',AUC_PR], run_eagerly = True)
        new_model.load_weights(checkpoint_path)

        X_test,y_test = data_generator(test, image_path, mask_path, height, width, channels, normalize = True,disable_bar=True)
        eval_result = new_model.evaluate(X_test,y_test,steps=1,return_dict = True) #get evaluation results
        eval_result['Kernel_size'] = starting_kernal_size

        output = output.append(eval_result, ignore_index=True)
        print(eval_result)

        output.to_csv('output.csv',index = False)

# count = 1 #counter for figures in for loops
# for image,mask in zip(X_test,y_test): #for loop for plotting images
    
#     img = image.reshape((1,height,width,channels))
#     pred_mask = new_model.predict(img)

#     plot_figures(image,pred_mask[:,:,:,-1], count, orig_mask=mask, ext = 'training')
#     plt.close('all')
#     count += 1

# plt.ioff()
# plt.show()
