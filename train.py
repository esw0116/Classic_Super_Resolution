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
        train_image_list = glob.glob('./dataset/training/gray_low/*.*')
        train_label_list = glob.glob('./dataset/training/gray/*.*')

        num_image = len(train_image_list)

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        v1, v2, prediction = sr_model.build_model()

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))

        train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, var_list=v1)
        train_op2 = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, var_list=v2)
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
                batch_image, batch_label = preprocess.load_data(train_image_list, train_label_list, j * batch_size,
                                                                (j + 1) * batch_size, self.patch_size, self.num_patch_per_image)

                mse_loss, _ = sess.run([loss, train_op], feed_dict={self.x: batch_image, self.y: batch_label})
                total_mse_loss += mse_loss/num_batch

            print('In', i+1, 'epoch, current loss is', total_mse_loss)
            saver.save(sess, save_path=self.save_path)

        print('Train completed')

    def train_vdsr(self, iteration):
        # images = low resolution, labels = high resolution
        sess = self.sess
        #load data
        train_image_list = glob.glob('./dataset/training/gray_low/*.*')
        train_label_list = glob.glob('./dataset/training/gray/*.*')

        num_image = len(train_image_list)

        sr_model = VDSR(channel_length=self.c_length, image=self.x)
        prediction, l2_loss = sr_model.build_model()

        learning_rate = tf.placeholder(dtype='float32', name='learning_rate')

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
            loss += 1e-4 * l2_loss
        optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # gradient clipping = Adam can handle by itself
        gvs = optimize.compute_gradients(loss=loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.1/learning_rate, 0.1/learning_rate), var) for grad, var in gvs]
        train_op = optimize.apply_gradients(capped_gvs)

        batch_size = 3
        num_batch = int(num_image/batch_size)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=5)
        if self.pre_trained:
            saver.restore(sess, self.save_path)

        lr = 0.1

        for i in range(iteration):
            total_mse_loss = 0
            total_l2 = 0
            if i % 20 == 19:
                lr = lr * 0.1

            for j in range(num_batch):
                batch_image, batch_label = preprocess.load_data(train_image_list, train_label_list, j * batch_size,
                                                                (j + 1) * batch_size, self.patch_size,
                                                                self.num_patch_per_image)
                '''
                train_image = np.array(Image.open(train_image_list[i]))
                train_image = train_image[np.newaxis, :, :, np.newaxis]
                train_label = np.array(Image.open(train_label_list[i]))
                train_label = train_label[np.newaxis, :, :, np.newaxis]
                residual = train_label - train_image
                '''

                l2, total_loss, _ = sess.run([l2_loss, loss, train_op], feed_dict={self.x: batch_image, self.y: batch_label, learning_rate: lr})
                total_mse_loss += total_loss/num_batch
                total_l2 += l2/num_batch

            print('In', '%04d' %(i+1), 'epoch, current loss is', '{:.5f}'.format(total_mse_loss), '{:.5f}'.format(total_l2))
            saver.save(sess, save_path=self.save_path)

        print('Train completed')