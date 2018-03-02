import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from model import SRCNN, VDSR


class TEST:
    def __init__(self, sess, channel_length, save_path):
        self.c_length = channel_length
        self.x = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.y = tf.placeholder(dtype='float32', shape=[None, None, None, self.c_length], name='image')
        self.save_path = save_path
        if sess is not None:
            self.sess = sess

    def test(self, mode, inference):
        # images = low resolution, labels = high resolution
        sess = self.sess

        # for training a particular image(one image)
        test_image_list_x2 = sorted(glob.glob('./dataset/test/X2/*.*'))
        test_image_list_x3 = sorted(glob.glob('./dataset/test/X3/*.*'))
        test_image_list_x4 = sorted(glob.glob('./dataset/test/X4/*.*'))
        test_label_list = sorted(glob.glob('./dataset/test/gray/*.*'))

        num_image = len(test_label_list)

        assert mode == 'SRCNN' or mode == 'VDSR'
        if mode == 'SRCNN':
            sr_model = SRCNN(channel_length=self.c_length, image=self.x)
            _, _, prediction = sr_model.build_model()
        elif mode == 'VDSR':
            sr_model = VDSR(channel_length=self.c_length, image=self.x)
            prediction, _ = sr_model.build_model()
            residual = prediction - self.x

        with tf.name_scope("PSNR"):
            psnr = 10 * tf.log(255 * 255 * tf.reciprocal(tf.reduce_mean(tf.square(self.y - prediction)))) / tf.log(tf.constant(10, dtype='float32'))

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        saver.restore(sess, self.save_path)

        avg_psnr = 0
        for i in range(num_image):
            test_image = np.array(Image.open(test_image_list_x2[i]))
            test_image = test_image[np.newaxis, :, :, np.newaxis]
            test_label = np.array(Image.open(test_label_list[i]))
            test_label = test_label[np.newaxis, :, :, np.newaxis]
            print(test_image.shape, test_label.shape)

            final_psnr = sess.run(psnr, feed_dict={self.x: test_image, self.y: test_label})

            print('X2 : Test PSNR is ', final_psnr)
            avg_psnr += final_psnr

            if inference:
                pred = sess.run(prediction, feed_dict={self.x: test_image, self.y: test_label})
                pred = np.squeeze(pred).astype(dtype='uint8')
                pred_image = Image.fromarray(pred)
                filename = './restored_vdsr/20180302/{}.png'.format(i)
                pred_image.save(filename)
                if mode == 'VDSR':
                    res = sess.run(residual, feed_dict={self.x: test_image, self.y: test_label})
                    res = np.squeeze(res).astype(dtype='uint8')
                    res_image = Image.fromarray(res)
                    filename = './restored_vdsr/20180302/{}_res.png'.format(i)
                    res_image.save(filename)

        print('X2 : Avg PSNR is ', avg_psnr/5)

        avg_psnr = 0
        for i in range(num_image):
            test_image = np.array(Image.open(test_image_list_x3[i]))
            h = test_image.shape[0]
            w = test_image.shape[1]
            h -= h % 3
            w -= w % 3
            test_image = test_image[np.newaxis, 0:h, 0:w, np.newaxis]
            test_label = np.array(Image.open(test_label_list[i]))
            test_label = test_label[np.newaxis, :, :, np.newaxis]

            final_psnr = sess.run(psnr, feed_dict={self.x: test_image, self.y: test_label})

            print('X3 : Test PSNR is ', final_psnr)
            avg_psnr += final_psnr

            if inference:
                pred = sess.run(prediction, feed_dict={self.x: test_image, self.y: test_label})
                pred = np.squeeze(pred).astype(dtype='uint8')
                pred_image = Image.fromarray(pred)
                filename = './restored_vdsr/20180302/{}.png'.format(i)
                pred_image.save(filename)
                if mode == 'VDSR':
                    res = sess.run(residual, feed_dict={self.x: test_image, self.y: test_label})
                    res = np.squeeze(res).astype(dtype='uint8')
                    res_image = Image.fromarray(res)
                    filename = './restored_vdsr/20180302/{}_res.png'.format(i)
                    res_image.save(filename)

        print('X3 : Avg PSNR is ', avg_psnr/5)

        avg_psnr = 0
        for i in range(num_image):
            test_image = np.array(Image.open(test_image_list_x4[i]))
            test_image = test_image[np.newaxis, :, :, np.newaxis]
            test_label = np.array(Image.open(test_label_list[i]))
            test_label = test_label[np.newaxis, :, :, np.newaxis]

            final_psnr = sess.run(psnr, feed_dict={self.x: test_image, self.y: test_label})

            print('X4 : Test PSNR is ', final_psnr)
            avg_psnr += final_psnr

            if inference:
                pred = sess.run(prediction, feed_dict={self.x: test_image, self.y: test_label})
                pred = np.squeeze(pred).astype(dtype='uint8')
                pred_image = Image.fromarray(pred)
                filename = './restored_vdsr/20180302/{}.png'.format(i)
                pred_image.save(filename)
                if mode == 'VDSR':
                    res = sess.run(residual, feed_dict={self.x: test_image, self.y: test_label})
                    res = np.squeeze(res).astype(dtype='uint8')
                    res_image = Image.fromarray(res)
                    filename = './restored_vdsr/20180302/{}_res.png'.format(i)
                    res_image.save(filename)

        print('X4 : Avg PSNR is ', avg_psnr/5)
