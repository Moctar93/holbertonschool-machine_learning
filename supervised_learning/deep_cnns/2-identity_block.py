#!/usr/bin/env python3
"""
Building the Inception Network as described in Going Deeper with Convolutions (2014).
"""

from tensorflow import keras as K

def inception_network():
    """
    Builds the Inception Network as described in Going Deeper with Convolutions (2014).
    
    Returns:
        model: The Keras model.
    """
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial layers
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception blocks
    inception_block = __import__('0-inception_block').inception_block
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # Final layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.4)(x)
    output_layer = K.layers.Dense(1000, activation='softmax')(x)

    model = K.models.Model(inputs=input_layer, outputs=output_layer)
    return model
