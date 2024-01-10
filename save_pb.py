#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Training script for local features.
"""

import os
import time
import yaml

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import training, testing
from preprocess import prepare_match_sets
from utils.npy_utils import EndPoints

from tools.common import Notify
from template.misc import summarizer
from template.solver import solver
from template.recoverer import recoverer, recoverer_test

FLAGS = tf.compat.v1.app.flags.FLAGS

# Params for config.
tf.compat.v1.app.flags.DEFINE_string('save_dir', None,
                           """Path to save the model.""")
tf.compat.v1.app.flags.DEFINE_string('train_config', None,
                           """Path to training configuration file.""")

def save(train_config):
    """The training procedure.
    Args:
    Returns:
        Nothing.
    """
    # Construct testing network.
    print(Notify.INFO, 'Construct testing network.', Notify.ENDC)
    endpoints = testing(train_config['network'])
    init_op = tf.compat.v1.global_variables_initializer()
    # GPU usage grows incrementally.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    with tf.compat.v1.Session(config=config) as sess:
        # Initialize variables.
        print(Notify.INFO, 'Running initialization operator.', Notify.ENDC)
        sess.run(init_op)
        recoverer_test(train_config['recoverer'], sess)
        save_path = os.path.join(FLAGS.save_dir, 'model.pb')

        # Freeze the graph
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            tf.compat.v1.get_default_graph().as_graph_def(),
            output_node_names=['l2norm'])

        # Save graph as protobuf
        with tf.compat.v1.gfile.GFile(save_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    with open(FLAGS.train_config, 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    save(train_config)


if __name__ == '__main__':
    tf.compat.v1.flags.mark_flags_as_required(['save_dir', 'train_config'])
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.app.run()
