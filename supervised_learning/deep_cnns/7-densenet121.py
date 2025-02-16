#!/usr/bin/env python3
"""
Building a DenseNet-121 architecture.
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Building a DenseNet-121 architecture.
    """
    input_data = K.Input(shape=(224, 224, 3))

    BN_0 = K.layers.BatchNormalization()(input_data)
    ReLU_0 = K.layers.Activation(activation='relu')(BN_0)

    conv_0 = K.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            kernel_initializer=K.initializers.he_normal(seed=0),
            padding="same")(ReLU_0)

    pool_0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=2,
                                   padding="same")(conv_0)

    out_1, nb = dense_block(pool_0, pool_0.shape[3], growth_rate, 6)
    trans_1, nb = transition_layer(out_1, nb, compression)

    out_2, nb = dense_block(trans_1, trans_1.shape[3], growth_rate, 12)
    trans_2, nb = transition_layer(out_2, nb, compression)

    out_3, nb = dense_block(trans_2, trans_2.shape[3], growth_rate, 24)
    trans_3, nb = transition_layer(out_3, nb, compression)

    out_4, nb = dense_block(trans_3, trans_3.shape[3], growth_rate, 16)

    avg_pooling = K.layers.AveragePooling2D(pool_size=(7, 7),
                                            padding="same")(out_4)

    dense = K.layers.Dense(units=1000,
                           activation='softmax')(avg_pooling)

    model = K.Model(input_data, dense)
    return model
