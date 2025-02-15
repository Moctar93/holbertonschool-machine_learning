#!/usr/bin/env python3
"""
Modified LeNet-5 Architecture
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def lenet5(x, y):
    """
    Function that builds a modified version of the LeNet-5 architecture using TensorFlow.

    Arguments:
    x -- tf.placeholder of shape (m, 28, 28, 1) containing the input images for the network
    y -- tf.placeholder of shape (m, 10) containing the one-hot labels for the network

    Returns:
    y_pred -- tensor for the softmax activated output
    train_op -- training operation that utilizes Adam optimization
    loss -- tensor for the loss of the network
    accuracy -- tensor for the accuracy of the network
    """
    # Initializer
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Layer 1: Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(inputs=x,
                             filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu)

    # Layer 2: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=2,
                                    strides=2)

    # Layer 3: Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu)

    # Layer 4: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=2,
                                    strides=2)

    # Flatten the output for the fully connected layers
    flatten = tf.layers.flatten(pool2)

    # Layer 5: Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(inputs=flatten,
                          units=120,
                          kernel_initializer=initializer,
                          activation=tf.nn.relu)

    # Layer 6: Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(inputs=fc1,
                          units=84,
                          kernel_initializer=initializer,
                          activation=tf.nn.relu)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    y_pred = tf.layers.dense(inputs=fc2,
                             units=10,
                             kernel_initializer=initializer,
                             activation=tf.nn.softmax)

    # Loss function
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred))

    # Optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return y_pred, train_op, loss, accuracy
