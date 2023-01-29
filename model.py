import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import trimesh


def conv_bn(x, filters):
    x = layers.Conv1D(filters, 1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


def dense_bn(x, n_units):
    x = layers.Dense(n_units)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):

    def __init__(self, n_features, l2reg=0.001):
        self.n_features = n_features
        self.l2reg = l2reg
        self.eye = tf.eye(n_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.n_features, self.n_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.n_features, self.n_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


class Tnet(tf.keras.layers.Layer):

    def __init__(self, n_features):
        super(Tnet, self).__init__()
        self.n_features = n_features
        self.bias = tf.constant_initializer(0.0)
        self.reg = OrthogonalRegularizer(n_features)
        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(32, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(64, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(512, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.global_max_pooling = layers.GlobalMaxPooling1D()

        self.dense1 = tf.keras.Sequential([
            layers.Dense(256),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.dense2 = tf.keras.Sequential([
            layers.Dense(128),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.dense3 = tf.keras.Sequential([
            layers.Dense(self.n_features * self.n_features,
                            kernel_regularizer=self.reg,
                            bias_initializer=self.bias),
            layers.Reshape((self.n_features, self.n_features))
        ])

        self.dt = layers.Dot(axes=(2, 1))

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_max_pooling(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dt([inputs, x])


class PointNet(tf.keras.Model):

    def __init__(self, num_class):
        super(PointNet, self).__init__()
        self.tnet1 = Tnet(3)
        self.tnet2 = Tnet(64)

        self.conv1 = tf.keras.Sequential([
            layers.Conv1D(64, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.conv2 = tf.keras.Sequential([
            layers.Conv1D(64, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.conv3 = tf.keras.Sequential([
            layers.Conv1D(64, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])

        self.max_pool = layers.GlobalMaxPooling1D()

        self.conv4 = tf.keras.Sequential([
            layers.Conv1D(128, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.conv5 = tf.keras.Sequential([
            layers.Conv1D(1024, 1, padding='valid'),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])

        self.dense1 = tf.keras.Sequential([
            layers.Dense(512),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])
        self.dense2 = tf.keras.Sequential([
            layers.Dense(256),
            layers.BatchNormalization(momentum=0.0),
            layers.Activation('relu')
        ])

        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        x = self.tnet1(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tnet2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.max_pool(x)

        x = self.dense1(x)
        x = self.dense2(x)

        x = self.dropout(x)
        x = self.classifier(x)

        return x


def build_model():
    model = PointNet()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

