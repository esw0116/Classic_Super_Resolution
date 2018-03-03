import tensorflow as tf
import glob
import numpy as np
from PIL import Image

from model import SRCNN, VDSR
from utils import preprocess



class TRAIN:
    def __init__(self, sess, channel_length, patch_size, save_path, pre_trained, num_patch_per_image):
        self.patch_size = patch_size
        self.num_patch_per_image = num_patch_per_image
        self.c_length = channel_length
        self.x = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.y = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.save_path = save_path
        self.pre_trained = pre_trained
        if sess is not None:
            self.sess = sess

    def train_srcnn(self, iteration):
        # images = low resolution, labels = high resolution
        sess = self.sess
        #load data
        train_image_list_x2 = glob.glob('./dataset/training/X2/*.*')
        train_image_list_x3 = glob.glob('./dataset/training/X3/*.*')
        train_image_list_x4 = glob.glob('./dataset/training/X4/*.*')
        train_label_list = glob.glob('./dataset/training/gray/*.*')

        num_image = len(train_label_list)

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        v1, v2, prediction = sr_model.build_model()

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))

        train_op1 = tf.train.GradientDescentOptimizer(learning_rate=5e-5).minimize(loss, var_list=v1)
        train_op2 = tf.train.GradientDescentOptimizer(learning_rate=5e-6).minimize(loss, var_list=v2)
        train_op = tf.group(train_op1, train_op2)

        # optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

        batch_size = 3
        num_batch = int(num_image/batch_size)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        if self.pre_trained:
            saver.restore(sess, self.save_path)

        for i in range(iteration):
            total_mse_loss = 0
            for j in range(num_batch):
                for k in range(3):
                    if k == 0:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x2, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)
                    if k == 1:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x3, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)
                    if k == 2:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x4, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)
                    mse_loss, _ = sess.run([loss, train_op], feed_dict={self.x: batch_image, self.y: batch_label})
                    total_mse_loss += mse_loss/(num_batch * 3)

            print('In', i+1, 'epoch, current loss is', total_mse_loss)
            saver.save(sess, save_path=self.save_path)

        print('Train completed')

    def train_vdsr(self, iteration):
        # images = low resolution, labels = high resolution
        sess = self.sess
        #load data
        train_image_list_x2 = sorted(glob.glob('./dataset/training/X2/*.*'))
        train_image_list_x3 = sorted(glob.glob('./dataset/training/X3/*.*'))
        train_image_list_x4 = sorted(glob.glob('./dataset/training/X4/*.*'))
        train_label_list = sorted(glob.glob('./dataset/training/gray/*.*'))

        num_image = len(train_label_list)

        sr_model = VDSR(channel_length=self.c_length, image=self.x)
        prediction, _, l2_loss = sr_model.build_model()

        learning_rate = tf.placeholder(dtype='float32', name='learning_rate')

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
            loss += 1e-4 * l2_loss

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # optimize = tf.train.AdamOptimizer(learning_rate=learning_rate, momentum=0.9)
        '''
        # gradient clipping = Adam can handle by itself
        gvs = optimize.compute_gradients(loss=loss)
        capped_gvs = [(tf.clip_by_value(grad, -10./learning_rate, 10./learning_rate), var) for grad, var in gvs]
        train_op = optimize.apply_gradients(capped_gvs)
        '''
        batch_size = 3
        num_batch = int((num_image - 1)/batch_size) + 1
        print(num_batch)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=2)
        if self.pre_trained:
            saver.restore(sess, self.save_path)

        lr = 1e-3

        for i in range(iteration):
            total_loss = 0 # mse + l2
            total_l2 = 0
            if i % 20 == 19:
                lr = lr * 0.9
            for j in range(num_batch):
                for k in range(3):
                    if k == 0:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x2, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)
                    if k == 1:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x3, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)
                    if k == 2:
                        batch_image, batch_label = preprocess.load_data(train_image_list_x4, train_label_list, j * batch_size,
                                                                        min((j + 1) * batch_size, num_image), self.patch_size,
                                                                        self.num_patch_per_image)

                    l2, losses, _ = sess.run([l2_loss, loss, train_op], feed_dict={self.x: batch_image, self.y: batch_label, learning_rate: lr})
                    total_loss += losses/(num_batch * 3)
                    total_l2 += 1e-4 * l2/(num_batch * 3)

            print('In', '%04d' %(i+1), 'epoch, current loss is', '{:.5f}'.format(total_loss - total_l2), '{:.5f}'.format(total_l2))
            saver.save(sess, save_path=self.save_path)

        print('Train completed')
