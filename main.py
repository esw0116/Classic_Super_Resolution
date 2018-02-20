import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from train import TRAIN
from test import TEST

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    channel_length = 1
    save_path = './save_model/gray_dataset'
    
    train_one_image = TRAIN(sess=sess, channel_length=channel_length, save_path=save_path, pre_trained=False, patch_size=33, num_patch_per_image=20)
    train_one_image.train(iteration=50000)
    '''
    infer_one_image = TEST(sess=sess, channel_length=channel_length, save_path=save_path)
    infer_one_image.test()
    '''
