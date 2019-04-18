#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:08:52 2019

@author: felix
"""

import numpy as np
import os
import tensorflow as tf
from util.util import get_dir, func_with_prob
from util.config import cfg


def _rand_crop_resize(min_size):
    def crop(img):
        crop_size = tf.random_uniform(shape=(), minval=min_size, maxval=1)
        x = tf.random_uniform(shape=(), minval=0, maxval=1-crop_size)
        y = tf.random_uniform(shape=(), minval=0, maxval=1-crop_size)
        box = [[x, y, x+crop_size, y+crop_size]]
        return tf.image.crop_and_resize([img], boxes=box, box_ind=[0], crop_size=[32,32])[0]

    return crop


def _chain_augs(*augs):
    def aug(x):
        for f in augs:
            x = f(x)
        x = tf.clip_by_value(x, 0, 1)
        return x
    return aug


def _aug(dataset, scale, prob):
    # Params for augmentation, between 0 (not at all) and 1
    param = {
        'angle': 0.25,
        'hue': 0.06,
        'sat': 0.4,
        'bright': 0.05,
        'contr': 0.3,
        'crop': 0.4
        }
    for key in param:
        param[key] *= scale

    tf.logging.debug('Building augmentation function with scale %f, prob %f', scale, prob)
    aug_func = func_with_prob(
        _chain_augs(
            lambda x: tf.contrib.image.rotate(x, param['angle'] * tf.random_normal(shape=())),
            tf.image.random_flip_left_right,
            lambda x: tf.image.random_hue(x, param['hue']),
            lambda x: tf.image.random_saturation(x, 1-param['sat'], 1+param['sat']),
            lambda x: tf.image.random_brightness(x, param['bright']),
            lambda x: tf.image.random_contrast(x, 1-param['contr'], 1+param['contr']),
            func_with_prob(_rand_crop_resize(1-param['crop']), 0.75)),
        prob)

    return dataset.map(lambda x, y: (aug_func(x), y), num_parallel_calls=32)


def to_tf_dataset(dataset, is_train=True, batch_size=None, aug=None):
    """
    Returns a shuffled tf dataset with the images and labels

    Args:
        dataset: ImgDataset instance
        is_train: If train or test data should be returned.
                  Train Data may be augmented
        batch_size: batch size of the dataset
        aug: Either a scalar augmentation strength,
             or a tuple (augmentation_strength, augmentation_probability)

    Returns:
        A tensorflow dataset
    """
    training, test = dataset.data
    x, y = training if is_train else test

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if is_train:
        if aug is not None and aug[0] is not None and aug[0] != 0:
            if not isinstance(aug, tuple):
                aug = (aug, 1.0)
            dataset = _aug(dataset, aug[0], aug[1])
        dataset = dataset.shuffle(4*batch_size if batch_size is not None else 1024)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


def get_attack_original(attack_name, dataset, n=None, targeted=False, override=False):
    """
    Loads original images from file, or creates file with random test images.
    Args:
        attack_name: string, name of the attack
        dataset: ImgDataset instance
        n: number of images. If image file already exists, None will load all.
           If file doesn't exists yet, or override is True, n must not be None
        override: If true, generate a new file
    """
    num_classes = dataset.num_classes
    path = get_dir(cfg.data_dir, dataset.name, attack_name)
    file_name = os.path.join(path, 'originals.npz')

    if not os.path.isfile(file_name) or override:
        _, (img, label) = dataset.data

        idx = np.random.permutation(len(img))
        img = img[idx]
        label = label[idx]

        img = img[:n]
        label = label[:n]

        save_kwargs = {'img': img, 'label': label}

        if targeted:
            target_label = np.random.randint(low=0, high=num_classes, size=label.size, dtype=label.dtype)
            # Make sure label and target label are different
            same_idx = np.where(label == target_label)[0]
            while same_idx.size > 0:
                target_label[same_idx] = np.random.randint(low=0, high=num_classes, size=same_idx.size, dtype=label.dtype)
                same_idx = np.where(label == target_label)[0]
            save_kwargs['target_label'] = target_label

        tf.logging.debug('Saving to file %s', file_name)
        np.savez(file_name, **save_kwargs)

    else:
        tf.logging.debug('Loading from file %s', file_name)
        with np.load(file_name) as npzfile:
            img = npzfile['img']
            label = npzfile['label']

            img = img[:n]
            label = label[:n]

            if 'target_label' in npzfile.keys():
                target_label = npzfile['target_label']
                target_label = target_label[:n]

    if targeted:
        return img, label, target_label
    return img, label
