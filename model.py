import numpy as np
from PIL import Image
import tensorflow as tf

class SRCNN:
    def __init__(self, channel_length, image):
        self.c_length = channel_length
        self.image = image

    def build_model(self):
        [c1, c2, c3] = [self.c_length, 64, 32]
        [f1, f2, f3] = [9, 1, 5]

        w1, b1 = self.get_weight_bias(f1, c1, c2)
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.image, w1, strides=[1,1,1,1], padding='SAME'), b1)
        relu1 = tf.nn.relu(conv1)
        w2, b2 = self.get_weight_bias(f2, c2, c3)
        conv2 = tf.nn.bias_add(tf.nn.conv2d(relu1, w2, strides=[1,1,1,1], padding='SAME'), b2)
        relu2 = tf.nn.relu(conv2)
        w3, b3 = self.get_weight_bias(f3, c3, c1)
        conv3 = tf.nn.bias_add(tf.nn.conv2d(relu2, w3, strides=[1, 1, 1, 1], padding='SAME'), b3)

        return conv3

    def get_weight_bias(self, filter_size, c_length1, c_length2):
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, c_length1, c_length2], dtype='float32',
                                                 stddev=1e-2), name='filter')
        bias = tf.Variable(tf.constant(0, shape=[c_length2], dtype='float32'), name='bias')
        return weight, bias

class VDSR:
    def __init__(self, channel_length, image):
        self.c_length = channel_length
        self.image = image

    def build_model(self):
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        conv = []
        conv.append(tf.layers.conv2d(self.image, 64, [3, 3], padding='SAME', activation=tf.nn.relu,
                                     kernel_regularizer=regularizer))

        for i in range(18):
            conv.append(tf.layers.conv2d(conv[i], 64, [3, 3], padding='SAME', activation=tf.nn.relu,
                                         kernel_regularizer=regularizer))

        conv_final = tf.layers.conv2d(conv[18], self.c_length, [3, 3], padding='SAME', activation=tf.nn.relu,
                                      kernel_regularizer=regularizer)
        l2_loss = tf.losses.get_regularization_loss()
        return conv_final, l2_loss
