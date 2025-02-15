#!/usr/bin/env python3
"""
Modified LeNet-5 Architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5 architecture
    using Keras.

    Arguments:
    X -- K.Input of shape (m, 28, 28, 1) containing the input images for
    the network

    Returns:
    model -- a K.Model compiled to use Adam optimization and accuracy
    metrics
    """
    # Initializer
    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Convolutional layer with 6 kernels of shape
    # 5x5 withsame padding
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding='same',
                            kernel_initializer=initializer,
                            activation='relu')(X)

    # Layer 2: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(pool_size=2,
                                  strides=2)(conv1)

    # Layer 3: Convolutional layer with 16 kernels of shape
    # 5x5 with valid padding
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding='valid',
                            kernel_initializer=initializer,
                            activation='relu')(pool1)

    # Layer 4: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(pool_size=2,
                                  strides=2)(conv2)

    # Flatten the output for the fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120,
                         kernel_initializer=initializer,
                         activation='relu')(flatten)

    # Layer 6: Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84,
                         kernel_initializer=initializer,
                         activation='relu')(fc1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(units=10,
                            kernel_initializer=initializer,
                            activation='softmax')(fc2)

    # Create the model
    model = K.models.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
