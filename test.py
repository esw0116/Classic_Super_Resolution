import tensorflow as tf
import numpy as np
from model import SRCNN
from .utils import preprocess

class TEST:
    def __init__(self, sess, channel_length, save_path):
        # self.image_size = image_size
        # self.label_size = label_size
        self.c_length = channel_length
        self.x = tf.placeholder([None, self.image_size, self.image_size, self.c_length], dtype='float32', name='image')
        self.y = tf.placeholder([None, self.label_size, self.label_size, self.c_length], dtype='float32', name='image')
        self.save_path = save_path
        if sess is not None:
            self.sess = sess

    def test(self):
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

        with tf.name_scope("PSNR"):
            psnr = 10 * tf.log(tf.reciprocal(tf.reduce_mean(tf.square(self.y - prediction)))) / tf.log(tf.constant(10))

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        saver.restore(sess, self.save_path)

        final_psnr = sess.run(psnr, feed_dict={self.x: image_y, self.y: label_y})

        print('Test PSNR is ', final_psnr)