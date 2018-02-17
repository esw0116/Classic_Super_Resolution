import tensorflow as tf
import numpy as np
from model import SRCNN
from .utils import preprocess


class TRAIN:
    def __init__(self, sess, channel_length, save_path):
        #self.image_size = image_size
        #self.label_size = label_size
        self.c_length = channel_length
        self.x = tf.placeholder([None, self.image_size, self.image_size, self.c_length], dtype='float32', name='image')
        self.y = tf.placeholder([None, self.label_size, self.label_size, self.c_length], dtype='float32', name='image')
        self.save_path = save_path
        if sess is not None:
            self.sess = sess

    def train(self, iteration):
        # images = low resolution, labels = high resolution
        sess = self.sess

        # for training a particular image(one image)
        image, label = preprocess.image_label_gen(image_path='sample/house_low.png', label_path='sample/house.png')
        image_y = image[:, :, 0]
        label_y = label[:, :, 0]

        # img size = label size = 525 * 680

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        prediction = sr_model.build_model()

        # pred size = 513 * 668
        label_y = label_y[6:519, 6:674]

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
        optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=5)

        for i in range(iteration):
            mse_loss, _ = sess.run([loss, optimize], feed_dict={self.x: image_y, self.y: label_y})

            if (i + 1) % 5 == 0:
                print('In', i+1, 'epoch, current loss is', mse_loss)
                saver.save(sess, save_path=self.save_path)

        print('Train completed')