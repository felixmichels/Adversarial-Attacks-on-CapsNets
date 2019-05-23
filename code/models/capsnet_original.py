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
import tfcaps as tc


class CapsNetOriginal(models.basicmodel.BasicModel):
    """
    The model from 'Dynamic Routing Between Capsules'
    """

    @lazy_scope_property
    def encoder(self):
        is_training = self.training
        i, o = tc.layers.new_io(self.img)

        i(tf.layers.conv2d(o(), filters=256, kernel_size=9, strides=1, activation=tf.nn.relu))

        tf.logging.debug('Shape after conv-layers: %s', o().get_shape())

        i(tc.layers.PrimaryConvCaps2D(kernel_size=9, types=32, dimensions=8, strides=2, data_format='channels_last'))

        tf.logging.debug('Shape after primary caps: %s', o().get_shape())

        i(tc.layers.ConvCaps2D(kernel_size=o().shape.as_list()[1:3], types=self.num_classes+self.garbage_class, dimensions=16, name="class-caps",
                               data_format='channels_last'))
        i(tf.squeeze(o(), axis=(1, 2)))  # shape: [batch, classes, dimensions]

        return o()

    @lazy_scope_property
    def probabilities(self):
        return tc.layers.length(self.encoder[:,:self.num_classes,:])

    @lazy_scope_property
    def logits(self):
        return 2*tf.atanh(2*self.probabilities - 1)

    @lazy_scope_property
    def decoder(self):
        encoder_out_masked_flat = tc.layers.label_mask(self.encoder[:,:self.num_classes,:], self.label, self.prediction, self.training)
        if self.garbage_class > 0:
            garbage_flat = tf.layers.flatten(self.encoder[:,-1,:])
            encoder_out_masked_flat = tf.concat([encoder_out_masked_flat, garbage_flat], axis=1)

        i, o = tc.layers.new_io(encoder_out_masked_flat)
        i(tf.layers.dense(o(), 512, activation=tf.nn.relu))
        i(tf.layers.dense(o(), 1024, activation=tf.nn.relu))
        i(tf.layers.dense(o(), np.multiply.reduce(self.shape), activation=tf.nn.sigmoid))
        i(tf.reshape(o(), [-1, *self.shape]))
        return o()

    @lazy_scope_property(only_training=True)
    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = opt.minimize(self.loss, global_step=tf.train.get_global_step())
        return train_op

    @lazy_scope_property(only_training=True)
    def summary_op(self):
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('l1_loss', self.l1_loss)
        tf.summary.scalar('l2_loss', self.l2_loss)
        return tf.summary.merge_all(scope=self.scope)

    @lazy_scope_property
    def l1_loss(self):
        reg = tf.contrib.layers.l1_regularizer(scale=self.l1_scale)
        weights = [v for v in tf.trainable_variables(self.scope) if 'bias' not in v.name]
        return tf.contrib.layers.apply_regularization(reg, weights)


    @lazy_scope_property
    def l2_loss(self):
        reg = tf.contrib.layers.l2_regularizer(scale=self.l2_scale)
        weights = [v for v in tf.trainable_variables(self.scope) if 'bias' not in v.name]
        return tf.contrib.layers.apply_regularization(reg, weights)

    @lazy_scope_property
    def recon_loss(self):
        return tc.losses.reconstruction_loss(original=self.img, reconstruction=self.decoder, alpha=self.recon_scale)

    @lazy_scope_property
    def loss(self):
        margin_loss = tc.losses.margin_loss(class_capsules=self.encoder, labels=self.label, m_minus=.1, m_plus=.9)
        return margin_loss + self.recon_loss + self.l1_loss + self.l2_loss
