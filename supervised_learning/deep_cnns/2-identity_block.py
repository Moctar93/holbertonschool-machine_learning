#!/usr/bin/env python3

import tensorflow.keras as K

def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for Image Recognition (2015).
    
    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple or list containing F11, F3, F12 respectively
    - s: stride of the first convolution in both the main path and the shortcut connection
    
    Returns:
    - Activated output of the projection block
    """
    F11, F3, F12 = filters
    he_initializer = K.initializers.HeNormal(seed=0)
    
    # First 1x1 Convolution (Reduce dimensionality)
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1,1), strides=(s,s), 
                             padding='valid', kernel_initializer=he_initializer)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    
    # 3x3 Convolution (Main feature extraction)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3,3), strides=(1,1), 
                             padding='same', kernel_initializer=he_initializer)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)
    
    # Second 1x1 Convolution (Expand dimensionality)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1,1), strides=(1,1), 
                             padding='valid', kernel_initializer=he_initializer)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)
    
    # Shortcut path (1x1 Convolution)
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1,1), strides=(s,s), 
                               padding='valid', kernel_initializer=he_initializer)(A_prev)
    shortcut_bn = K.layers.BatchNormalization(axis=3)(shortcut)
    
    # Add shortcut to main path
    add = K.layers.Add()([bn3, shortcut_bn])
    output = K.layers.Activation('relu')(add)
    
    return output
