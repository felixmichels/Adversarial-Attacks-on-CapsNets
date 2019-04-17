#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:55:04 2019

@author: felix
"""

import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from util.config import cfg
from util.lazy import lazy_property


class ImgDataset(object):

    def __init__(self, dataloader, name, num_classes, label_names=None, scale=True):
        """
        Args:
            dataloader: Function, that returns data in the form of
                        tuples (train_x, train_y), (test_x, test_y).
                        First axis is assumed to be sample number.
                        Will only be called once
            name: string, name of the dataset
            num_classes: Number of classes
            label_names: Array with names corresponding to the labels
            scale: If true, data will be scaled from range [0, 255] to [0, 1]
        """

        self._loader = dataloader
        self.num_classes = num_classes
        self._scale = scale
        self.label_names = label_names
        self.name = name

    @lazy_property
    def data(self):
        (train_x, train_y), (test_x, test_y) = self._loader()
        train_x = train_x.astype('float32')
        train_y = train_y.squeeze().astype('int64')

        test_x = test_x.astype('float32')
        test_y = test_y.squeeze().astype('int64')

        assert len(train_x) == len(train_x), 'Training data size not matching'
        assert len(test_x) == len(test_y), 'Test data size not matching'
        assert train_y.shape[1:] == test_y.shape[1:] == (), 'Invalid label shape'
        assert train_x.shape[1:] == test_x.shape[1:], 'Test and training data not matching'
        
        if self._scale:
            train_x /= 255.0
            test_x /= 255.0

        return (train_x, train_y), (test_x, test_y)

    @lazy_property
    def num_train(self):
        return len(self.data[0][0])

    @lazy_property
    def num_test(self):
        return len(self.data[1][0])

    @lazy_property
    def shape(self):
        return self.data[0][0].shape




cifar10 = ImgDataset(
    tf.keras.datasets.cifar10.load_data,
    name='cifar10',
    num_classes=10,
    label_names=['airplane',
                 'automobile',
                 'bird',
                 'cat',
                 'deer',
                 'dog',
                 'frog',
                 'horse',
                 'ship',
                 'truck'
                 ])


fashion_mnist = ImgDataset(
    tf.keras.datasets.fashion_mnist.load_data,
    name='fashion_mnist',
    num_classes=10,
    label_names=['t-shirt',
                'trouser',
                'pullover',
                'dress',
                'coat',
                'sandal'
                'shirt'
                'sneaker'
                'bag'
                'ankle boot'
                ])


mnist = ImgDataset(
    tf.keras.datasets.mnist.load_data,
    name='mnist',
    num_classes=10,
    label_names=[str(n) for n in range(10)])


def _svhn_loader():
    train = sio.loadmat(os.path.join(cfg.datset_dir, 'train_32x32.mat'))
    test = sio.loadmat(os.path.join(cfg.datset_dir, 'test_32x32.mat'))
    # Change from HWCN to NHWC
    train_x = np.roll(train['X'], 3, 0)
    test_x = np.roll(test['X'], 3, 0)
    return (train_x, train['y']), (test_x, test['y'])


svhn = ImgDataset(
    _svhn_loader,
    name='svhn',
    num_classes=10,
    label_names=[str(n) for n in range(10)])


_datasets = {ds.name: ds for ds in (cifar10, fashion_mnist, mnist, svhn)}


def dataset_by_name(name):
    """ Gets dataset object with corresponding name """
    return _datasets[name.lower()]
