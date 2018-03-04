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
    parser.add_argument('--model')
    parser.add_argument('--phase')
    args = parser.parse_args()
    save_path = args.save_path
    model = args.model
    phase = args.phase
    # save_path = './save_model/20170222/srcnn'

    if phase == 'train':
        train_images = TRAIN(sess=sess, channel_length=channel_length, save_path=save_path, pre_trained=True, patch_size=33, num_patch_per_image=25)
        if model == 'SRCNN':
            train_images.train_srcnn(iteration=5000)
        elif model == 'VDSR':
            train_images.train_vdsr(iteration=100)

    elif phase == 'test':
        infer_one_image = TEST(sess=sess, channel_length=channel_length, save_path=save_path)
        infer_one_image.test(mode=model, inference=True)
