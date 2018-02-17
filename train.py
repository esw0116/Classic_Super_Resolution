import tensorflow as tf
import numpy as np
from model import SRCNN

class TRAIN:
    def __init__(self, sess, image_size, label_size, channel_length):
        self.image_size = image_size
        self.label_size = label_size
        self.c_length = channel_length
        self.x = tf.placeholder([None, self.image_size, self.image_size, self.c_length], dtype='float32', name='image')
        self.y = tf.placeholder([None, self.label_size, self.label_size, self.c_length], dtype='float32', name='image')

        if sess is not None:
            self.sess = sess

    def train(self):
        sess = self.sess

        sr_model = SRCNN(image_size=self.image_size, label_size=self.label_size, channel_length=self.c_length,
                         image=self.x)
        prediction = sr_model.build_model()

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
        optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

