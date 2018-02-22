import numpy as np
import math
import tensorflow as tf


def he_normal(seed=None, scale=1.0):
    """
    He Normal initializer
    Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing human-level
    performance on imagenet classification. arXiv preprint arXiv:1502.01852.

    Args:
        scale: float
               Scaling factor for the weights. Set this to ``1.0`` for linear and
               sigmoid units, to ``sqrt(2)`` for rectified linear units, and
               to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
               leakiness ``alpha``. Other transfer functions may need different factors.
    """
    return tf.contrib.layers.variance_scaling_initializer(factor=2.0 * scale, mode='FAN_IN',
                                                          uniform=False, seed=seed, dtype=tf.float32)


class SRCNN:
    def __init__(self, channel_length, image):
        self.c_length = channel_length
        self.image = image

    def get_weight_bias(self, filter_size, c_length1, c_length2, name):
        weight = tf.Variable(tf.random_normal(shape=[filter_size, filter_size, c_length1, c_length2], stddev=1e-3),
                             name=name + '_filter')
        bias = tf.Variable(tf.constant(0, shape=[c_length2], dtype='float32'), name=name + '_bias')
        return weight, bias

    def build_model(self):
        [c1, c2, c3] = [self.c_length, 64, 32]
        [f1, f2, f3] = [9, 1, 5]

        w1, b1 = self.get_weight_bias(f1, c1, c2, name='Conv1')
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.image, w1, strides=[1,1,1,1], padding='SAME'), b1)
        relu1 = tf.nn.relu(conv1)
        w2, b2 = self.get_weight_bias(f2, c2, c3, name='Conv2')
        conv2 = tf.nn.bias_add(tf.nn.conv2d(relu1, w2, strides=[1,1,1,1], padding='SAME'), b2)
        relu2 = tf.nn.relu(conv2)
        w3, b3 = self.get_weight_bias(f3, c3, c1, name='Conv3')
        conv3 = tf.nn.bias_add(tf.nn.conv2d(relu2, w3, strides=[1, 1, 1, 1], padding='SAME'), b3)

        return [w1, b1, w2, b2], [w3, b3], conv3

class VDSR:
    def __init__(self, channel_length, image):
        self.c_length = channel_length
        self.image = image

    def get_weight_bias(self, filter_size, c_length1, c_length2, name, scale=math.sqrt(2)):
        weight = tf.get_variable(shape=[filter_size, filter_size, c_length1, c_length2], name=name+'_filter',
                                 initializer=he_normal(scale=scale))
        bias = tf.Variable(tf.constant(0, shape=[c_length2], dtype='float32'), name=name + '_bias')
        return weight, bias

    def build_model(self):
        w = [None] * 20
        b = [None] * 20
        conv = [None] * 20
        relu = [None] * 19
        l2_loss = 0

        w[0], b[0] = self.get_weight_bias(3, self.c_length, 64, name='Conv0')
        conv[0] = tf.nn.bias_add(tf.nn.conv2d(self.image, w[0], strides=[1,1,1,1], padding='SAME'), b[0])
        relu[0] = tf.nn.relu(conv[0])
        l2_loss += tf.nn.l2_loss(w[0])
        for i in range(1, 19):
            w[i], b[i] = self.get_weight_bias(3, 64, 64, name='Conv'+str(i))
            conv[i] = tf.nn.bias_add(tf.nn.conv2d(relu[i-1], w[i], strides=[1,1,1,1], padding='SAME'), b[i])
            relu[i] = tf.nn.relu(conv[i])
            l2_loss += tf.nn.l2_loss(w[i])
        w[19], b[19] = self.get_weight_bias(3, 64, self.c_length, name='Conv19', scale=1.)
        conv[19] = tf.nn.bias_add(tf.nn.conv2d(relu[18], w[19], strides=[1,1,1,1], padding='SAME'), b[19])
        l2_loss += tf.nn.l2_loss(w[19])

        conv_fin = tf.add(conv[19], self.image)

        return conv_fin, l2_loss
        '''
        regularizer = tf.contrib.layers.l2_regularizer(1e-4)
        conv = []
        conv.append(tf.layers.conv2d(self.image, 64, [3, 3], padding='SAME', activation=tf.nn.relu,
                                     kernel_regularizer=regularizer))

        for i in range(18):
            conv.append(tf.layers.conv2d(conv[i], 64, [3, 3], padding='SAME', activation=tf.nn.relu,
                                         kernel_regularizer=regularizer))

        conv_final = tf.layers.conv2d(conv[18], self.c_length, [3, 3], padding='SAME', activation=None,
                                      kernel_regularizer=regularizer)
        l2_loss = tf.losses.get_regularization_loss()

        conv_final += self.image
        return conv_final, l2_loss
        '''
