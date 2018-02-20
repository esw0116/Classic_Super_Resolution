import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from model import SRCNN
from utils import preprocess

class TEST:
    def __init__(self, sess, channel_length, save_path):
        self.c_length = channel_length
        self.x = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.y = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.save_path = save_path
        if sess is not None:
            self.sess = sess

    def test(self, inference):
        # images = low resolution, labels = high resolution
        sess = self.sess

        # for training a particular image(one image)
        test_image_list = glob.glob('./dataset/test/gray_low/*.*')
        test_label_list = glob.glob('./dataset/test/gray/*.*')

        num_image = len(test_image_list)

        # img size = label size = 525 * 680

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        prediction = sr_model.build_model()

        with tf.name_scope("PSNR"):
            psnr = 10 * tf.log(255 * 255 * tf.reciprocal(tf.reduce_mean(tf.square(self.y - prediction)))) / tf.log(tf.constant(10, dtype='float32'))

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        saver.restore(sess, self.save_path)
        for i in range(num_image):
            test_image = np.array(Image.open(test_image_list[i]))
            test_image = test_image[np.newaxis, :, :, np.newaxis]
            test_label = np.array(Image.open(test_label_list[i]))
            test_label = test_label[np.newaxis, :, :, np.newaxis]

            final_psnr = sess.run(psnr, feed_dict={self.x: test_image, self.y: test_label})

            print('Test PSNR is ', final_psnr)

            if inference:
                pred = sess.run(prediction, feed_dict={self.x: test_image, self.y: test_label})
                pred = np.squeeze(pred).astype(dtype='uint8')
                pred_image = Image.fromarray(pred)
                pred_image.show()