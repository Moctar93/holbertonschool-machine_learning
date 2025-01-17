#!/usr/bin/env python3
"""
Module to build, train, and save a neural network classifier.
"""
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.
    
    :param X_train: numpy.ndarray, training input data
    :param Y_train: numpy.ndarray, training labels (one-hot encoded)
    :param X_valid: numpy.ndarray, validation input data
    :param Y_valid: numpy.ndarray, validation labels (one-hot encoded)
    :param layer_sizes: list, containing the number of nodes in each layer
    :param activations: list, containing the activation functions for each layer
    :param alpha: float, learning rate
    :param iterations: int, number of iterations to train over
    :param save_path: string, path where the model should be saved
    
    :return: string, the path where the model was saved
    """
    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Create forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss
    loss = calculate_loss(y_pred, y)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_pred, y)

    # Create the training operation (optimizer)
    train_op = create_train_op(loss, alpha)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # Create a session to run the graph
    with tf.Session() as sess:
        sess.run(init)

        # Train over iterations
        for i in range(iterations + 1):
            # Run the training step (feed the data for training)
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            # Every 100 iterations, 0th iteration, and last iteration, print details
            if i == 0 or i % 100 == 0 or i == iterations:
                # Calculate the cost and accuracy for training and validation
                train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
                valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

                # Print the results
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the trained model to the specified path
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path

