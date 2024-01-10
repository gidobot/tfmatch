#!/usr/bin/env python3
"""
Copyright 2018, Zixin Luo, HKUST.
The script defining the recoverer.
"""

import tensorflow as tf

from tools.common import Notify


def optimistic_restore(session, save_file):
    reader = tf.compat.v1.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.compat.v1.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.compat.v1.global_variables()), tf.compat.v1.global_variables()))
    with tf.compat.v1.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.compat.v1.train.Saver(restore_vars)
    saver.restore(session, save_file)

def recoverer_test(config, sess):
    """
    Recovery parameters from a pretrained model.
    Args:
        config: The recoverer configuration.
        sess: The tensorflow session instance.
    Returns:
        step: The step value.
    """
    if config['pretrained_model'] is not None and config['ckpt_step'] is not None:
        optimistic_restore(sess, config['pretrained_model'])
        print(Notify.INFO, 'Pre-trained model restored from %s' %
              config['pretrained_model'], Notify.ENDC)
    else:
        print(Notify.WARNING, 'No checkpoint specified.', Notify.ENDC)

def recoverer(config, sess):
    """
    Recovery parameters from a pretrained model.
    Args:
        config: The recoverer configuration.
        sess: The tensorflow session instance.
    Returns:
        step: The step value.
    """
    if config['pretrained_model'] is not None and config['ckpt_step'] is not None:
        restore_var = []
        # selectively recover the parameters.
        if config['exclude_var'] is None:
            restore_var = tf.compat.v1.global_variables()
        else:
            keyword = config['exclude_var'].split(',')
            for tmp_var in tf.compat.v1.global_variables():
                find_keyword = False
                for tmp_keyword in keyword:
                    if tmp_var.name.find(tmp_keyword) >= 0:
                        print(Notify.WARNING, 'Ignore the recovery of variable',
                              tmp_var.name, Notify.ENDC)
                        find_keyword = True
                        break
                if not find_keyword:
                    restore_var.append(tmp_var)
        try:
            restorer = tf.compat.v1.train.Saver(restore_var)
            restorer.restore(sess, config['pretrained_model'])
        except Exception as e:
            print(e)
            print("Using optimistic restore")
            optimistic_restore(sess, config['pretrained_model'])
        print(Notify.INFO, 'Pre-trained model restored from %s' %
              config['pretrained_model'], Notify.ENDC)
        return config['ckpt_step']
    else:
        print(Notify.WARNING, 'Training from scratch.', Notify.ENDC)
        return 0
