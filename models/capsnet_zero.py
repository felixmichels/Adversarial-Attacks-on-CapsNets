#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:01:04 2019

@author: felix
"""

import tensorflow as tf
import numpy as np
import models.basicmodel
from util.lazy import lazy_scope_property
from util.config import cfg
import tfcaps as tc


class CapsNet(models.basicmodel.BasicModel):
    """
    First try for a cifar10 capsule net
    """

    @property
    def name(self):
        return 'CapsNetZero'

    @lazy_scope_property
    def encoder(self):
        """
        Define encoder part
        :param inputs: Inputs for the encoder
        :param classes: Number of classes
        :return:
        """
        is_training = self.train_placeholder
        i, o = tc.layers.new_io(self.img)

        i(tf.layers.batch_normalization(
            tf.layers.conv2d(o(), filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
            training=is_training))
        i(tf.layers.batch_normalization(
            tf.layers.conv2d(o(), filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
            training=is_training))
        i(tf.layers.batch_normalization(
            tf.layers.conv2d(o(), filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
            training=is_training))
        i(tf.layers.batch_normalization(
            tf.layers.conv2d(o(), filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
            training=is_training))

        i(tf.concat([o(-1), o(-2), o(-3), o(-4), o(-5)], axis=-1))
        tf.logging.debug('Shape after conv-layers: %s', o().get_shape())

        i(tc.layers.PrimaryConvCaps2D(kernel_size=9, types=32, dimensions=8, strides=1, data_format='channels_last'))
        i(tc.layers.ConvCaps2D(kernel_size=3, types=64, dimensions=12, strides=2, name='conv-caps0',
                               data_format='channels_last'))
        i(tc.layers.ConvCaps2D(kernel_size=11, types=cfg.classes, dimensions=16, name="conv-caps",
                               data_format='channels_last'))
        i(tf.squeeze(o(), axis=(1, 2)))  # shape: [batch, classes, dimensions]

        return o()

    @lazy_scope_property
    def probabilities(self):
        return tc.layers.length(self.encoder)

    @lazy_scope_property
    def decoder(self):
        """
        Define decoder part
        :param inputs: Inputs for the decoder.
        :param shape: Shape of a single data point. For MNIST it would be [28, 28, 1].
        :return:
        """
        encoder_out_masked_flat = tc.layers.label_mask(self.encoder, self.label, self.prediction, self.train_placeholder)

        i, o = tc.layers.new_io(encoder_out_masked_flat)
        i(tf.layers.dense(o(), 512, activation=tf.nn.relu))
        i(tf.layers.dense(o(), 1024, activation=tf.nn.relu))
        i(tf.layers.dense(o(), np.multiply.reduce([32,32,3]), activation=tf.nn.sigmoid))
        i(tf.reshape(o(), [-1, 32, 32, 3]))
        return o()

    @lazy_scope_property
    def prediction(self):
        return tf.argmax(self.probabilities, axis=-1)

    @lazy_scope_property(only_training=True)
    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = opt.minimize(self.loss, global_step=tf.train.get_global_step())
        return train_op

    @lazy_scope_property(only_training=True)
    def summary(self):
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Loss', self.loss)

    @lazy_scope_property
    def accuracy(self):
        correct_preds = tf.equal(self.prediction, self.label)
        return  tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.size(self.label), tf.float32)

    @lazy_scope_property
    def loss(self):
        margin_loss = tc.losses.margin_loss(class_capsules=self.encoder, labels=self.label, m_minus=.1, m_plus=.9)
        recon_loss = tc.losses.reconstruction_loss(original=self.img, reconstruction=self.decoder, alpha=0.0005)
        tf.summary.scalar('recon_loss', recon_loss)
        l2_loss = 1e-6 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        tf.summary.scalar('l2_loss', l2_loss)
        return margin_loss + recon_loss + l2_loss
