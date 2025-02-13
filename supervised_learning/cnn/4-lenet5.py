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


# Example code to run the model and print the predictions
if __name__ == "__main__":
    with tf.Session() as sess:
        # Example input data (replace with actual data)
        x_input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))  # Example shape
        y_input = tf.placeholder(tf.float32, shape=(None, 10))          # Example shape

        # Build model
        y_pred, train_op, loss_op, accuracy_op = lenet5(x_input, y_input)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Simulate input data (use actual dataset here)
        x_data = [[[[0]]]]  # Dummy data, replace with actual input
        y_data = [[0] * 10]  # Dummy labels, replace with actual labels

        # Run the model (no actual training here)
        feed_dict = {x_input: x_data, y_input: y_data}
        predictions = sess.run(y_pred, feed_dict=feed_dict)

        # Print predictions
        print(predictions)

