import tensorflow as tf
from train import TRAIN
from test import TEST

with tf.Session() as sess:
    channel_length = 1
    save_path = './save_model/one_image'
    '''
    train_one_image = TRAIN(sess=sess, channel_length=channel_length, save_path=save_path, pre_trained=True)
    train_one_image.train(iteration=50000)
    '''
    infer_one_image = TEST(sess=sess, channel_length=channel_length, save_path=save_path)
    infer_one_image.test()
