#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior to use 1.x

def lenet5(x, y):
    # Initialize variables and activation function
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    # 1st Convolutional layer
    conv1 = tf.layers.conv2d(inputs=x, filters=6, kernel_size=(5, 5), padding="same",
                             activation=activation, kernel_initializer=init)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2))

    # 2nd Convolutional layer
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=(5, 5), padding="valid",
                             activation=activation, kernel_initializer=init)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2))

    # Flatten layer
    flatten = tf.layers.flatten(pool2)

    # Fully connected layers
    fc1 = tf.layers.dense(inputs=flatten, units=120, activation=activation,
                          kernel_initializer=init)
    fc2 = tf.layers.dense(inputs=fc1, units=84, activation=activation,
                          kernel_initializer=init)
    fc3 = tf.layers.dense(inputs=fc2, units=10, activation=None,
                          kernel_initializer=init)

    # Prediction (softmax)
    y_pred = tf.nn.softmax(fc3)

    # Loss (softmax cross-entropy)
    loss = tf.losses.softmax_cross_entropy(y, fc3)

    # Accuracy
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # Training operation
    train = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train, loss, mean

