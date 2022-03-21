import enum
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Activation, Add
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import training
from utils.DropBlock import DropBlock2D
from skimage import measure
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy 
from skimage.morphology import disk
import albumentations as A
import tensorflow as tf
import numpy as np
import shutil
import random
import glob
import sys
import cv2
import os

def dynamic_unet_cnn(height,width,channels,num_layers = 4,starting_filter_size = 16, use_dropout = False, dropsize = 0.9, blocksize = 7,num_classes = 1,final_activation = 'sigmoid'): #Unet-cnn model 
    inputs = Input((height, width, channels))
    s = inputs

    for i in range(num_layers):
        if i == 0:
            curr_filter_size = starting_filter_size
            # print(curr_filter_size)

            conv = Conv2D(curr_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same') (s)
            conv = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv, training = use_dropout)
            # conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            conv = Conv2D(curr_filter_size, (3,3), activation = None, kernel_initializer = 'he_normal', padding = 'same')(conv)
            conv = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv, training = use_dropout)
            # conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            pool = MaxPooling2D((2,2))(conv)

            conv_list = list([conv])
            pool_list = list([pool])

        else: 
            curr_filter_size = curr_filter_size*2
            # print(curr_filter_size)

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same')(pool_list[i-1]))
            conv_list[i] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[i], training = use_dropout)
            # conv_list[i] = BatchNormalization()(conv_list[i])
            conv_list[i] = Activation('relu')(conv_list[i])

            conv_list[i] = Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (conv_list[i])
            conv_list[i] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[i], training = use_dropout)
            # conv_list[i] = BatchNormalization()(conv_list[i])
            conv_list[i] = Activation('relu')(conv_list[i])
            
            pool = MaxPooling2D((2, 2)) (conv_list[i])

            pool_list.append(pool)

    curr_filter_size = curr_filter_size*2
    # print(curr_filter_size)

    conv_list_reverse = conv_list.copy()
    conv_list_reverse.reverse()
    
    conv_list.append(Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (pool_list[num_layers-1]))
    conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
    # conv_list[-1] = BatchNormalization()(conv_list[-1])
    conv_list[-1] = Activation('relu')(conv_list[-1])

    conv_list[-1] = spatial_attention_block(conv_list[-1])

    ##### spatial attention block goes here
    conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (conv_list[-1])
    conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
    # conv_list[-1] = BatchNormalization()(conv_list[-1])
    conv_list[-1] = Activation('relu')(conv_list[-1])

    for i in range(num_layers):
        if i == 0:
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u = Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1])
            # u = concatenate([u, conv_list_reverse[i]])
            u = Add()([u, conv_list_reverse[i]])

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u))
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (conv_list[-1])
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

            u_list = list([u])

        elif i == (num_layers-1):
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            # u_list[i] = concatenate([u_list[i], conv_list_reverse[i]],axis=3)
            u_list[i] = Add()([u_list[i], conv_list_reverse[i]])

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u_list[i]))
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (conv_list[-1])
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

        else: 
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            # u_list[i] = concatenate([u_list[i], conv_list_reverse[i]])
            u_list[i] = Add()([u_list[i], conv_list_reverse[i]])

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (u_list[i]))
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation=None, kernel_initializer='he_normal', padding='same') (conv_list[-1])
            # conv_list[-1] = DropBlock2D(keep_prob = dropsize, block_size = blocksize)(conv_list[-1], training = use_dropout)
            # conv_list[-1] = BatchNormalization()(conv_list[-1])
            conv_list[-1] = Activation('relu')(conv_list[-1])

    outputs = Conv2D(num_classes, (1, 1), activation=final_activation) (conv_list[-1])
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return(model)

def spatial_attention_block(input_features):

    avg_pool = tf.math.reduce_mean(input_features, axis = -1, keepdims=True)
    max_pool = tf.math.reduce_max(input_features, axis = -1,keepdims=True)

    cat_pools = concatenate([avg_pool,max_pool])

    convolved = Conv2D(1, (7,7), activation='sigmoid', kernel_initializer='he_normal', padding='same')(cat_pools)

    output = tf.math.multiply(convolved,input_features)

    return output

def dynamic_wnet_cnn(height,width,channels,num_layers = 4,starting_filter_size = 16, use_dropout = False, dropsize = 0.9, blocksize = 7,num_classes = 1):

    model1 = dynamic_unet_cnn(height,width,channels,
        num_layers = num_layers, starting_filter_size = starting_filter_size, use_dropout = True, num_classes=num_classes)

    model2 = dynamic_unet_cnn(height,width,num_classes,
        num_layers = num_layers, starting_filter_size = starting_filter_size, use_dropout = True, num_classes = num_classes)

    Wnet_model = tf.keras.Sequential()
    Wnet_model.add(model1)
    Wnet_model.add(model2)

    return Wnet_model

def plot_figures(image,pred_mask,num, orig_mask = None,ext = '', epoch = None, sup_title = None, BGR = False): #function for plotting figures

    if ext != '':
        output_path = os.path.join(os.getcwd(),'output_images' + '_' + ext)
    else:
        output_path = os.path.join(os.getcwd(),'output_images')

    if BGR is True:
        image = image[:,:, ::-1]

    try:
        os.mkdir(output_path)
    except:
        pass

    if orig_mask is not None:
        plt.figure(num,figsize=(12,12))
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Image")
        plt.subplot(132)
        plt.imshow(orig_mask.squeeze(),cmap='gray')
        plt.title("Original Mask")
        plt.subplot(133)
        plt.imshow(pred_mask.squeeze(),cmap='gray')
        plt.title('Predicted Mask')
    else:
        plt.figure(num,figsize=(12,12))
        plt.figure(num)
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Image")
        plt.subplot(122)
        plt.imshow(pred_mask.squeeze(),cmap='gray')
        plt.title('Predicted Mask')

    if epoch is not None:
        output_name = os.path.join(output_path,str(epoch) + '_' + str(num) + '.png')
    else:
        output_name = os.path.join(output_path,str(num) + '.png')

    if sup_title is not None:
        plt.suptitle(sup_title)

    plt.savefig(output_name, dpi = 96*2)

def plot_acc_loss(results): #plot accuracy and loss
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
        
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

def data_generator(dataset, image_path, mask_path, height, width, channels, create_more_data = None, data_multiplication = 2, normalize = False,shuffle_data = False,disable_bar=False): #function for generating data
    
    if disable_bar is False:
        print('Loading in training data')

    X_train = np.zeros((len(dataset),height,width,channels), dtype = np.uint8) #initialize training sets (and testing sets)
    y_train = np.zeros((len(dataset),height,width,1), dtype = np.uint8)

    X_train = np.zeros((len(dataset),height,width,channels)) #initialize training sets (and testing sets)
    y_train = np.zeros((len(dataset),height,width,1))

    sys.stdout.flush() #write everything to buffer ontime 

    for i in tqdm(range(len(dataset)),total=len(dataset),disable=disable_bar): #iterate through datatset and build X_train,y_train

        new_image_path = os.path.join(image_path,dataset.iloc[i][0])
        new_mask_path = os.path.join(mask_path,dataset.iloc[i][1])

        if channels == 1:
            image = cv2.imread(new_image_path,0)
            image = np.expand_dims(image,axis = -1)
        else:
            image = cv2.imread(new_image_path)[:,:,::-1]
        mask = cv2.imread(new_mask_path)[:,:,:1]

        img_resized = cv2.resize(image,(height,width))
        mask_resized = cv2.resize(mask,(height,width),interpolation = cv2.INTER_NEAREST)

        # mask_resized = np.expand_dims(mask_resized,axis=2)

        img_resized = np.atleast_3d(img_resized).astype(np.float64)
        mask_resized = np.atleast_3d(mask_resized).astype(np.float64)

        # img_resized = resize(image,(height,width), mode = 'constant',preserve_range = True)
        # mask_resized = resize(mask, (height,width), mode = 'constant', preserve_range = True)

        X_train[i] = img_resized
        y_train[i] = mask_resized

    if create_more_data is not None:
        X_train_noise = X_train.astype(np.uint8)
        y_train_noise = y_train.astype(np.uint8)

        for i in range(data_multiplication - 1):

            transform = A.Compose([
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.GridDropout(p=0.5),
                A.augmentations.transforms.RandomBrightnessContrast(p=0.75),
                A.augmentations.transforms.ChannelShuffle(p=0.5),
                A.augmentations.transforms.GaussNoise(p=0.33),
                A.augmentations.transforms.RandomGridShuffle(p=0.5)
                ])

            if disable_bar is False:
                print('Augmenting dataset for additional data')

            for j in tqdm(range(len(dataset)),total=len(dataset),disable=disable_bar):

                transformed = transform(image=X_train[j].astype(np.uint8), mask=y_train[j].astype(np.uint8))
                X_train_noise[j] = transformed['image']
                y_train_noise[j] = transformed['mask']

            X_train_noise = X_train_noise.astype(np.float64)
            y_train_noise = y_train_noise.astype(np.float64)

            X_train = np.concatenate((X_train,X_train_noise),axis = 0)
            y_train = np.concatenate((y_train,y_train_noise),axis = 0)

            if shuffle_data:
                shuffle_idx = np.arange(len(y_train))
                np.random.shuffle(shuffle_idx)

                X_train = X_train[shuffle_idx]
                y_train = y_train[shuffle_idx]

    if normalize:
        if disable_bar is False:
            print('Normalizing input data to 3*std per rgb')
        for count in tqdm(range(len(X_train)),total=len(X_train),disable=disable_bar):
            X_train[count] = normalize_each_RGB_subset(X_train[count])
            y_train[count] = y_train[count] / 255

    return X_train, y_train

def load_first_image_get_size(img_path,dataset = None, force_img_size = None):

    if force_img_size is not None:

        img_size = force_img_size

    else:
        if dataset is not None:
            img = cv2.imread(os.path.join(img_path,dataset['images'][0]),0)
            img_size = np.min(img.shape)
        else:
            img_paths = natsorted(glob.glob(os.path.join(img_path,'*.png')))
            if img_paths:
                img = cv2.imread(img_paths[0],0)
                img_size = np.min(img.shape)
            else:
                print('Could not find any Pngs using jpgs and tifs instead')
                img_paths = natsorted(glob.glob(os.path.join(img_path,'*.png'))) + natsorted(glob.glob(os.path.join(img_path,'*.jpg'))) + natsorted(glob.glob(os.path.join(img_path,'*.tif')))
                img = cv2.imread(img_paths[0],0)
                img_size = np.min(img.shape)

    return img_size

def get_num_layers_unet(img_size):

    power_of_2 = round(np.log(img_size)/np.log(2)) # round to closest 2^x

    new_img_size = 2**power_of_2 # get to closest power of 2 

    # the last layer will be 16x16xDepth 
    # can change to other sizes, inital unet used a 32x32xDepth 
    num_layers = int(np.log(new_img_size/32)/np.log(2))

    return num_layers, new_img_size

def data_generator_for_testing(image_path, height = None, width = None,channels = None, num_to_load = None, recursive = False, spcific_file_ext = 'png',normalize = False,disable_bar=False): #function for generating data

    if recursive:
        dataset = natsorted(glob.glob(os.path.join(image_path,'**/*.' + spcific_file_ext),recursive = True))
    else:
        dataset = natsorted(glob.glob(os.path.join(image_path,'*.png')))
        if dataset == []:
            dataset = natsorted(glob.glob(os.path.join(image_path,'*.jpg'))) 
            if dataset == []:
                dataset = natsorted(glob.glob(os.path.join(image_path,'*.tif')))

    if num_to_load is not None:
        dataset = random.sample(dataset,num_to_load)

    if height is None or width is None or channels is None:
        ex_img = cv2.imread(dataset[0])

        height = ex_img.shape[0]
        width = ex_img.shape[1]

        if len(ex_img.shape) > 2:
            max_img = np.max(ex_img,axis=-1)
            mean_img = np.mean(ex_img,axis=-1)

            if np.sum(mean_img - max_img) == 0:
                channels = 1
            else:
                channels = 3
        else:
            channels = 1

    images = [] #initialize training sets (and testing sets)

    sys.stdout.flush() #write everything to buffer ontime 
    if disable_bar is False:
        print('Loading in data')

    for i in tqdm(range(len(dataset)),total=len(dataset),disable=disable_bar):

        this_img_path = dataset[i]
        
        if channels == 1:
            image = cv2.imread(this_img_path,0)
        else:
            image = cv2.imread(this_img_path)

        if normalize:
            image = normalize_each_RGB_subset(image)

        img_resized = cv2.resize(image,(height,width))
        images.append(np.atleast_3d(img_resized))

    return images

class test_on_improved_val_loss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        curr_val_loss = logs['val_AUC_PR']
        try:
            val_loss_hist = self.model.history.history['val_AUC_PR']
        except:
            val_loss_hist = curr_val_loss + 1

        if epoch == 0:
            try:
                os.mkdir(os.path.join(os.getcwd(), 'output_images_testing_during'))
            except:
                shutil.rmtree(os.path.join(os.getcwd(), 'output_images_testing_during'))
                os.mkdir(os.path.join(os.getcwd(), 'output_images_testing_during'))

        if curr_val_loss >= np.max(val_loss_hist) or epoch == 0:

            test_path = os.path.join(os.getcwd(), 'testing')

            if os.path.isdir(test_path):

                model_shape = self.model.input_shape[-3:]
                test_height = model_shape[0]
                test_width = model_shape[1]
                test_depth = model_shape[2]
                test_imgs = data_generator_for_testing(test_path, height=test_height,width=test_width,channels=test_depth,normalize=True,disable_bar=True)

                for count, img in enumerate(test_imgs):

                    img_reshape = np.expand_dims(img,axis = 0)

                    in_img = img_reshape.astype(np.float64)

                    pred_mask = self.model.predict(in_img)

                    if count == 0:
                        concat_in_imgs = in_img.squeeze()
                        concat_out_imgs = pred_mask[:,:,:,-1].squeeze()
                    else:
                        concat_in_imgs = cv2.vconcat([concat_in_imgs,in_img.squeeze()])
                        concat_out_imgs = cv2.vconcat([concat_out_imgs,pred_mask[:,:,:,-1].squeeze()])

                plot_figures(concat_in_imgs,concat_out_imgs, count, ext = 'testing_during', epoch = epoch, sup_title = curr_val_loss, BGR = True)
                plt.close('all')

def bwareafilt(mask, n=1, area_range=(0, np.inf)):


    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas

def bwareaopen(img, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of 
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)
    
    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0
            
    return img

def normalize_each_RGB_subset(img):

    img_shape = img.shape

    output = np.zeros(shape = img.shape)

    # if RGB
    if len(img_shape) > 2:
        for i in range(img_shape[-1]):

            this_slice = img[:,:,i]

            nonzero_vals = this_slice[np.nonzero(this_slice)]

            mean_val = np.mean(nonzero_vals)
            std_val = np.std(nonzero_vals)

            # define normalized value as 95% confidence interval per color
            if (mean_val + 3*std_val) > 255:
                norm_value = 255
            else:
                norm_value = (mean_val + 3*std_val)

            norm_slice = this_slice / norm_value
            norm_slice = np.clip(norm_slice,0,1)

            output[:,:,i] = norm_slice
    # if grayscale
    else:
        mean_val = np.mean(img)
        std_val = np.std(img)

        # define normalized value as 95% confidence interval per color
        if (mean_val + 2*std_val) > 255:
            norm_value = 255
        else:
            norm_value = (mean_val + 2*std_val)

        norm_slice = img / norm_value
        norm_slice = np.clip(norm_slice,0,1)

        output = norm_slice

    # out2 = output.copy()
    # for i in range(3):
    #     out2[:,:,i] = entropy(output[:,:,i],disk(3))
    #     out2[:,:,i] = out2[:,:,i]/np.max(out2[:,:,i])

    return output

def insert_text_into_img(img,text):

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5,30)
    fontScale = 1
    color_white = (1, 1, 1)
    color_black = (0,0,0)

    thickness_black = 3
    thickness_white = 1

    image_black = cv2.putText(img, text, org, font, 
                   fontScale, color_black, thickness_black, cv2.LINE_AA)

    out_img = cv2.putText(image_black, text, org, font, 
                   fontScale, color_white, thickness_white, cv2.LINE_AA)

    return out_img