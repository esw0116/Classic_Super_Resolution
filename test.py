import tensorflow as tf
import numpy as np
from PIL import Image
from model import SRCNN
from utils import preprocess

class TEST:
    def __init__(self, sess, channel_length, save_path):
        # self.image_size = image_size
        # self.label_size = label_size
        self.c_length = channel_length
        self.x = tf.placeholder(dtype='float32', shape=[None, 525, 680, self.c_length], name='image')
        self.y = tf.placeholder(dtype='float32', shape=[None, 525, 680, self.c_length], name='image')
        self.save_path = save_path
        if sess is not None:
            self.sess = sess

    def test(self):
        # images = low resolution, labels = high resolution
        sess = self.sess

        # for training a particular image(one image)
        image, label = preprocess.image_label_gen(image_path='sample/house_low.png', label_path='sample/house.png')
        image_y = image[:, :, 0]

        image_y = image_y[np.newaxis, :, :, np.newaxis]
        label_y = label[:, :, 0]
        label_y = label_y[np.newaxis, :, :, np.newaxis]

        # img size = label size = 525 * 680

        sr_model = SRCNN(channel_length=self.c_length, image=self.x)
        prediction = sr_model.build_model()

        with tf.name_scope("PSNR"):
            psnr = 10 * tf.log(255 * 255 * tf.reciprocal(tf.reduce_mean(tf.square(self.y - prediction)))) / tf.log(tf.constant(10, dtype='float32'))

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        saver.restore(sess, self.save_path)

        final_psnr = sess.run(psnr, feed_dict={self.x: image_y, self.y: label_y})

        print('Test PSNR is ', final_psnr)

        pred_y = sess.run(prediction, feed_dict={self.x: image_y, self.y: label_y})
        pred_y = np.squeeze(pred_y).astype(dtype='uint8')
        image_cb = image[:, :, 1]
        image_cr = image[:, :, 2]

        original =np.stack((image_y, image_cb, image_cr), axis=-1)
        orig_img = Image.fromarray(original, mode='YCbCr')
        orig_img.show()

        restore = np.stack((pred_y, image_cb, image_cr), axis=-1)
        print(restore.shape)
        img = Image.fromarray(restore, mode='YCbCr')
        img.show()
        img2 = img.convert('RGB')
        img2.save('house_sr.png')
