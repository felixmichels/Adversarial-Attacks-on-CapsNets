#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:36:43 2019

@author: felix
"""

import tensorflow as tf
import models.basicmodel
from util.lazy import lazy_scope_property
from tfcaps.layers import new_io


class SimpleNet(models.basicmodel.BasicModel):

    @lazy_scope_property
    def logits(self):
        i, o = new_io(self.img)
        i(tf.layers.flatten(self.img))
        i(2 * (o() - 1))

        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.l1_scale,
            scale_l2=self.l2_scale)

        i(tf.layers.dense(o(), self.num_classes, kernel_regularizer=regularizer))
        return o()

    @lazy_scope_property
    def probabilities(self):
        return tf.nn.softmax(self.logits)

    @lazy_scope_property(only_training=True)
    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        return opt.minimize(self.loss, global_step=tf.train.get_global_step())
    
    @lazy_scope_property(only_training=True)
    def summary(self):
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Regularization', self.regularization_loss)
        
    @lazy_scope_property
    def loss(self):
        entropy = tf.nn.softmax_cross_entropy(self.one_hot_label, self.logits)

        return tf.reduce_mean(entropy, name='loss') + self.regularization_loss

    @lazy_scope_property
    def regularization_loss(self):
        return tf.losses.get_regularization_loss()
