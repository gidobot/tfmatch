#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Training script for local features.
"""

import os
import time
import yaml
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from model_keras import training_dataset
from preprocess import prepare_match_sets
from utils.npy_utils import EndPoints
from losses import KerasLoss

from tools.common import Notify
from template.misc import summarizer
from template.solver import solver
from template.recoverer import recoverer

from cnn_wrapper.desckeras import DescNet

FLAGS = flags.FLAGS

# Params for config.
flags.DEFINE_string('save_dir', None,
                           """Path to save the model.""")
flags.DEFINE_string('gl3d', None,
                           """Path to dataset root.""")
flags.DEFINE_integer('num_corr', 1024,
                            """The correspondence number of one sample.""")
# Training config
flags.DEFINE_string('train_config', None,
                           """Path to training configuration file.""")
flags.DEFINE_string('data_split', 'comb',
                           """Which data split in GL3D will be used.""")
flags.DEFINE_boolean('is_training', None,
                            """Flag to training model.""")
flags.DEFINE_boolean('regenerate', False,
                            """Flag to re-generate training samples.""")
flags.DEFINE_boolean('dry_run', False,
                            """Whether to enable dry-run mode in data generation (useful for debugging).""")
flags.DEFINE_integer('device_idx', 0,
                            """GPU device index.""")
flags.DEFINE_integer('batch_size', 2,
                            """Training batch size.""")

# class TBCallback(tf.keras.callbacks.TensorBoard):
#     def __init__(self, log_every=1, **kwargs):
#         super().__init__(**kwargs)
#         self.log_every = log_every
#         self.counter = 0
    
#     def on_train_batch_end(self, batch, logs=None):
#         self.counter+=1
#         if self.counter%self.log_every==0:
#             for name, value in logs.items():
#                 if name in ['batch', 'size']:
#                     continue
#                 print("logging {}".format(name))
#                 summary = tf.summary.scalar(name, value)
#                 self._train_writer.add_summary(summary, self.counter)
#             self._train_writer.flush()
        
#         super().on_train_batch_end(batch, logs)


def train(sample_list, img_list, depth_list, reg_feat_list, train_config):
    """The training procedure.
    Args:
        sample_list: List of training sample file paths.
        img_list: List of image paths.
        depth_list: List of depth paths.
        reg_feat_list: List of regional features.
    Returns:
        Nothing.
    """
    # Construct training networks.
    print(Notify.INFO, 'Running on GPU indexed', FLAGS.device_idx, Notify.ENDC)
    print(Notify.INFO, 'Construct training networks.', Notify.ENDC)

    # Instantiate an SGD optimizer with exponential learning rate decay.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        train_config['solver']['lr']['base_lr'],
        decay_steps=train_config['solver']['lr']['stepvalue'],
        decay_rate=train_config['solver']['lr']['gamma'],
        staircase=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                        momentum=train_config['solver']['optimizer']['momentum'],
                                        weight_decay=train_config['solver']['regularization']['weight_decay'])

    # Define input tensors
    input0 = tf.keras.Input(shape=(FLAGS.num_corr, 32, 32, 1), batch_size=FLAGS.batch_size, name='input0')
    input1 = tf.keras.Input(shape=(FLAGS.num_corr, 32, 32, 1), batch_size=FLAGS.batch_size, name='input1')

    # Instantiate feature towers
    feat_tower0 = DescNet().build(input0)
    feat_tower1 = DescNet().build(input1)

    # Instantiate a loss layer.
    inlier_mask_input = tf.keras.Input(shape=(FLAGS.num_corr), batch_size=FLAGS.batch_size, name='input_mask')
    feat0 = tf.reshape(feat_tower0.output, (FLAGS.batch_size, FLAGS.num_corr, 128))
    feat1 = tf.reshape(feat_tower1.output, (FLAGS.batch_size, FLAGS.num_corr, 128))
    loss_feat0_layer, loss_feat1_layer, loss = KerasLoss(loss_type='LOG')(feat0, feat1, inlier_mask_input)

    tf.summary.scalar('loss', loss)

    # siamese style training
    siamese = tf.keras.Model(inputs=(feat_tower0.inputs, feat_tower1.inputs, inlier_mask_input), outputs=(loss_feat0_layer, loss_feat1_layer))

    # siamese = training(sample_list, img_list, depth_list, reg_feat_list, train_config['network'])

    # compile model
    dump_config = train_config['dump']
    # siamese.compile(optimizer=optimizer, steps_per_execution=dump_config['display'])
    siamese.compile(optimizer=optimizer)
    siamese.summary()

    # Save the model checkpoint periodically.
    checkpoint_path = os.path.join(FLAGS.save_dir, 'model-{epoch:02d}.hdf5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1,
                                                 save_freq=dump_config['snapshot'])

    # tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./log",
                                                    update_freq='epoch')
                                                    # update_freq=dump_config['display'])
    # tensorboard_callback = TBCallback(log_dir="./log", log_every=dump_config['display'])

    # train model
    dataset = training_dataset(sample_list, img_list, depth_list, reg_feat_list, train_config['network'])
    epochs = dump_config['max_steps'] // dump_config['display']
    siamese.fit(dataset,
            epochs=epochs,
            steps_per_epoch=dump_config['display'],
            callbacks=[cp_callback, tensorboard_callback])
            # callbacks=[tensorboard_callback])

def unit_test(sample_list, img_list, depth_list, reg_feat_list, train_config):
    dataset = training_dataset(sample_list, img_list, depth_list, reg_feat_list, train_config['network'])

    for data in dataset:
        cv2.namedWindow("test1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("test2", cv2.WINDOW_NORMAL)
        cv2.imshow('test1', data['input0'][0,0,:,:].numpy())
        cv2.imshow('test2', data['input1'][0,0,:,:].numpy())
        cv2.waitKey()

def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    flags.mark_flags_as_required(['is_training', 'gl3d', 'train_config'])
    if FLAGS.is_training:
        flags.mark_flags_as_required(['save_dir'])

    with open(FLAGS.train_config, 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    # Prepare training samples.
    sample_list, img_list, depth_list, reg_feat_list = prepare_match_sets(
        regenerate=FLAGS.regenerate, is_training=FLAGS.is_training, data_split=FLAGS.data_split)
    # Training entrance.
    train(sample_list, img_list, depth_list, reg_feat_list, train_config)
    # unit_test(sample_list, img_list, depth_list, reg_feat_list, train_config)

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    app.run(main)
