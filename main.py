import argparse
import tensorflow as tf
from train import TRAIN
from test import TEST

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    channel_length = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    args = parser.parse_args()
    save_path = args.save_path
    # save_path = './save_model/20170222/srcnn'
    
    train_images = TRAIN(sess=sess, channel_length=channel_length, save_path=save_path, pre_trained=False, patch_size=33, num_patch_per_image=20)
    train_images.train_vdsr(iteration=100)
    '''
    infer_one_image = TEST(sess=sess, channel_length=channel_length, save_path=save_path)
    infer_one_image.test(mode='SRCNN', inference=True)
    '''
