import tensorflow as tf
import glob
from model import SRCNN, VDSR
from utils import preprocess


class TRAIN:
    def __init__(self, sess, channel_length, patch_size, save_path, pre_trained, num_patch_per_image):
        self.patch_size = patch_size
        self.num_patch_per_image = num_patch_per_image
        self.c_length = channel_length
        self.x = tf.placeholder(dtype='float32', shape=[None, self.patch_size, self.patch_size, self.c_length], name='image')
        self.y = tf.placeholder(dtype='float32', shape=[None, self.patch_size, self.patch_size, self.c_length], name='image')
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
        print(num_image)

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        prediction = sr_model.build_model()

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
        optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        batch_size = 3
        num_batch = int(num_image/batch_size)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=5)
        if self.pre_trained:
            saver.restore(sess, self.save_path)

        for i in range(iteration):
            total_mse_loss = 0
            for j in range(num_batch):
                batch_image, batch_label = preprocess.load_data(train_image_list, train_label_list, j * batch_size,
                                                                (j + 1) * batch_size, self.patch_size, self.num_patch_per_image)

                mse_loss, _ = sess.run([loss, optimize], feed_dict={self.x: batch_image, self.y: batch_label})
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
        print(num_image)

        sr_model = VDSR(channel_length=self.c_length, image=self.x)
        prediction = sr_model.build_model()

        with tf.name_scope("mse_loss"):
            loss = tf.reduce_mean(tf.square(self.y - prediction))
        optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        batch_size = 3
        num_batch = int(num_image/batch_size)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=5)
        if self.pre_trained:
            saver.restore(sess, self.save_path)

        for i in range(iteration):
            total_mse_loss = 0
            for j in range(num_batch):
                batch_image, batch_label = preprocess.load_data(train_image_list, train_label_list, j * batch_size,
                                                                (j + 1) * batch_size, self.patch_size, self.num_patch_per_image)
                batch_residual = batch_label - batch_image

                mse_loss, _ = sess.run([loss, optimize], feed_dict={self.x: batch_image, self.y: batch_residual})
                total_mse_loss += mse_loss/num_batch

            print('In', i+1, 'epoch, current loss is', total_mse_loss)
            saver.save(sess, save_path=self.save_path)

        print('Train completed')