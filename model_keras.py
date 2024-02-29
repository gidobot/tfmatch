#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Model architectures.
"""
import tensorflow as tf
import numpy as np
import cv2
import sys
import threading
import matplotlib.pyplot as plt

from losses import make_structured_loss, make_quadruple_loss, make_detector_loss
from utils.tf_utils import apply_patch_pert, apply_coord_pert, photometric_augmentation
from utils.tf_geom import get_warp, get_dist_mat, interpolate
from utils.npy_utils import get_rnd_homography, get_rnd_affine

from tools.common import Notify
from tools.io import load_pfm
from cnn_wrapper import helper
import cnn_wrapper.spatial_transformer as st

from cnn_wrapper.desckeras import DescNet
from cnn_wrapper.descnet import GeoDesc

from absl import flags

FLAGS = flags.FLAGS

DB_IMG_SIZE = 1000

lock = threading.Lock()

def training_dataset(match_set_list, img_list, depth_list, reg_feat_list, config):
    """Queue to read training data in binary.
    Args:
        spec: Model specifications.
        match_set_list: List of samples.
        img_list: List of image paths.
        depth_list: List of depth paths.
        reg_feat_list: List of reginal features.
    Returns:
        batch_tensors: List of fetched data.
    """

    spec = helper.get_data_spec(model_class=GeoDesc)

    # sample queue. the sample list has been shuffled.
    def _match_set_parser(val):
        def _parse_img(img_paths, idx):
            img_path = tf.squeeze(tf.gather(img_paths, idx))
            img = tf.image.decode_image(
                tf.io.read_file(img_path), channels=1)
            img.set_shape((DB_IMG_SIZE, DB_IMG_SIZE, 1))
            if config['resize'] > 0:
                img = tf.image.resize(
                    img, (config['resize'], config['resize']))
                pad_size = int(config['resize'] * 0.1)
            else:
                pad_size = int(DB_IMG_SIZE * 0.1)
            if not config['dense_desc']:
                # avoid boundary effect.
                img = tf.pad(img, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                             mode='SYMMETRIC')
            return img

        def _parse_depth(depth_paths, idx):
            depth = tf.numpy_function(
                load_pfm, [tf.squeeze(tf.gather(depth_paths, idx))], tf.float32)
            depth.set_shape((DB_IMG_SIZE // 4, DB_IMG_SIZE // 4))
            target_size = DB_IMG_SIZE // 4
            if config['resize'] > 0:
                target_size = config['resize']
            if target_size != DB_IMG_SIZE // 4:
                depth = tf.image.resize(
                    depth[..., None], (target_size, target_size))
                depth = tf.squeeze(depth, axis=-1)
            return depth

        def _parse_reg_feat(reg_feat_paths, idx, reg_feat_reso, reg_feat_dim):
            reg_feat_path = tf.squeeze(tf.gather(reg_feat_paths, idx))
            reg_feat = tf.io.decode_raw(
                tf.io.read_file(reg_feat_path), tf.float32)
            reg_feat = tf.reshape(
                reg_feat, (reg_feat_reso, reg_feat_reso, reg_feat_dim))
            return reg_feat

        def _preprocess(img, kpt_coeff, spec, num_corr, photaug, pert_homo, pert_affine, dense_desc):
            """
            Data Preprocess.
            """
            img = tf.cast(img, tf.float32)
            # img.set_shape((spec.batch_size,
            #                img.get_shape()[1],
            #                img.get_shape()[2],
            #                img.get_shape()[3]))
            img = tf.expand_dims(img, 0)
            if FLAGS.is_training and photaug:
                print(Notify.WARNING, 'Applying photometric augmentation.', Notify.ENDC)
                # img = tf.map_fn(photometric_augmentation, img, back_prop=False)
                img = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(photometric_augmentation, img))
            img = tf.clip_by_value(img, 0, 255)
            # perturb patches and coordinates.
            pert_kpt_affine, kpt_ncoords = apply_patch_pert(
                kpt_coeff, pert_affine, 1, num_corr, adjust_ratio=1. if dense_desc else 5. / 6.)
                # kpt_coeff, pert_affine, spec.batch_size, num_corr, adjust_ratio=1. if dense_desc else 5. / 6.)
            # patch sampler.
            patch = st.transformer_crop(
                img, pert_kpt_affine, spec.input_size, True)
            # patch standardization
            mean, variance = tf.nn.moments(patch, axes=[1, 2], keepdims=True)
            out = tf.nn.batch_normalization(
                patch, mean, variance, None, None, 1e-5)
            out = tf.stop_gradient(out)
            # image warp for SIFT compute
            img = tf.cast(img, tf.uint8)

            return out, kpt_ncoords, pert_homo, img

        decoded = tf.io.decode_raw(val, tf.float32)
        idx0 = tf.cast(decoded[0], tf.int32)
        idx1 = tf.cast(decoded[1], tf.int32)
        inlier_num = tf.cast(decoded[2], tf.int32)
        ori_img_size0 = tf.reshape(decoded[3:5], (2,))
        ori_img_size1 = tf.reshape(decoded[5:7], (2,))
        K0 = tf.reshape(decoded[7:16], (3, 3))
        K1 = tf.reshape(decoded[16:25], (3, 3))
        e_mat = tf.reshape(decoded[25:34], (3, 3))
        rel_pose = tf.reshape(decoded[34:46], (3, 4))
        kpt_coeff0 = tf.slice(decoded, [46], [6 * 1024])
        kpt_coeff1 = tf.slice(
            decoded, [46 + 6 * 1024], [6 * 1024])
        # parse images.
        img0 = _parse_img(img_list, idx0)
        img1 = _parse_img(img_list, idx1)
        # parse depths
        depth0 = _parse_depth(depth_list, idx0)
        depth1 = _parse_depth(depth_list, idx1)
        # inlier mask
        inlier_mask = tf.concat([tf.ones(inlier_num), tf.zeros(1024 - inlier_num)], axis=0)
        inlier_mask = tf.cast(inlier_mask, tf.bool)
        # generate random affine/homography transformations
        pert_homo = tf.numpy_function(get_rnd_homography, [2, 1, 0.15], tf.float32)
            # get_rnd_homography, [2, spec.batch_size, 0.15], tf.float32)
        # pert_homo = tf.reshape(pert_homo, (2, spec.batch_size, 3, 3))
        pert_homo = tf.reshape(pert_homo, (2, 1, 3, 3))

        pert_affine = tf.numpy_function(get_rnd_affine, [2, 1, 1024], tf.float32)
            # get_rnd_affine, [2, spec.batch_size, num_corr], tf.float32)
        # pert_affine = tf.reshape(pert_affine, (2, spec.batch_size, num_corr, 3, 3))
        pert_affine = tf.reshape(pert_affine, (2, 1, 1024, 3, 3))
        # preprocess
        net_input0, kpt_ncoords0, pert_homo0, img_aug0 = _preprocess(
            img0, kpt_coeff0, spec, 1024, config['photaug'],
            pert_homo[0], pert_affine[0], config['dense_desc'])
        net_input1, kpt_ncoords1, pert_homo1, img_aug1 = _preprocess(
            img1, kpt_coeff1, spec, 1024, config['photaug'],
            pert_homo[1], pert_affine[1], config['dense_desc'])

        # fetch_tensors = [img0, img1, depth0, depth1, kpt_coeff0, kpt_coeff1, inlier_num,
                         # ori_img_size0, ori_img_size1, K0, K1, e_mat, rel_pose]
        # fetch_tensors = [tf.squeeze(net_input0), tf.squeeze(net_input1), inlier_mask]

        # indices = tf.range(start=0, limit=tf.shape(net_input0)[0], dtype=tf.int32)
        # shuffled_indices = tf.random.shuffle(indices)[:FLAGS.num_corr]
        # net_input0 = tf.gather(tf.squeeze(net_input0), shuffled_indices, axis=0)
        # net_input1 = tf.gather(tf.squeeze(net_input1), shuffled_indices, axis=0)
        # inlier_mask = tf.gather(inlier_mask, shuffled_indices, axis=0)

        net_input0 = tf.squeeze(net_input0)
        net_input1 = tf.squeeze(net_input1)

        fetch_tensors = {'input0': net_input0, 'input1': net_input1, 'input_mask': inlier_mask}
        return fetch_tensors

    def _batch_merge(val):
        val['input0'] = tf.reshape(val['input0'], (2048, 32, 32, 1))
        val['input1'] = tf.reshape(val['input1'], (2048, 32, 32, 1))
        val['input_mask'] = tf.reshape(val['input_mask'], (-1,1))
        return val

    # def _interlieve(val):
    #     x = val.numpy()
    #     shape = x.shape
    #     ig = np.zeros((2*shape[0] - 1, 32,  32, 1))
    #     idx = list(range(0, ig.shape[0], 2))
    #     ig[idx] = val
    #     return ig

    def _gridify(val):
        i0 = val['input0']
        i1 = val['input1']
        ind = tf.expand_dims(tf.range(0, 2*2048-1, 2), -1)
        ig0 = tf.scatter_nd(indices=ind, updates=i0, shape=tf.constant([2*2048-1, 32, 32, 1]))
        ig1 = tf.scatter_nd(indices=ind, updates=i1, shape=tf.constant([2*2048-1, 32, 32, 1]))
        # shape = i0.shape
        # ig0 = np.zeros((2*shape[0] - 1, 32,  32, 1))
        # ig1 = np.zeros((2*shape[0] - 1, 32,  32, 1))
        # idx = list(range(0, ig.shape[0], 2))
        # ig0[idx] = i0
        # ig1[idx] = i1
        # val['input0'] = tf.reshape(ig0, (1, -1, 32, 1))
        # val['input1'] = tf.reshape(ig1, (1, -1, 32, 1))
        # ig0 = _interlieve(val['input0'])
        # ig1 = _interlieve(val['input1'])
        val['input0'] = tf.reshape(ig0, (1, -1, 32, 1)) 
        val['input1'] = tf.reshape(ig1, (1, -1, 32, 1)) 
        return val


    # decoded:
    # [1] inlier_num: 1 float
    # [2] idx: 2 float
    # [3] ori_img_size0: 2 float
    # [4] ori_img_size1: 2 float
    # [5] K0: 9 float
    # [6] K1: 9 float
    # [7] e_mat: 9 float
    # [8] rel_pose: 12 float
    # [9] kpt_coeff: 1024 * 6 * 2 float
    # [10] geo_sim: 1024 float
    dataset = tf.data.FixedLengthRecordDataset(match_set_list, 53432)
    # dataset = tf.data.FixedLengthRecordDataset(match_set_list, 4*(46 + num_corr*13))
    if FLAGS.is_training:
        dataset = dataset.shuffle(buffer_size=spec.batch_size * 32)
    dataset = dataset.repeat()
    dataset = dataset.map(
        _match_set_parser, num_parallel_calls=spec.batch_size * 2)
    dataset = dataset.batch(spec.batch_size)
    dataset = dataset.map(_batch_merge)
    if config['grid_mode']:
        dataset = dataset.map(_gridify)
    dataset = dataset.prefetch(buffer_size=spec.batch_size * 4)
    # dataset = dataset.prefetch(buffer_size=spec.batch_size * 4)
    # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # batch_tensors = iterator.get_next()
    # return batch_tensors
    return dataset