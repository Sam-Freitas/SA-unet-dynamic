# Dynamic spatial anntentuation Unet masking neural network

This is an implementation of the Unet CNN with 3 major tweaks

1 - the model dynamically adjusts the depth of the network for the specific size of the images

2 - during the final convolutional step-down there is a spatial attentuation block 

3 - usage of random dropblocks instead of random dropouts 

---------------------------------------

The goal of this is to create a easily changable and powerful Masking Neural Network for people to use
