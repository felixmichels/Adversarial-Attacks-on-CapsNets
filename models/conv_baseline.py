#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:35:34 2019

@author: felix
"""

import tensorflow as tf
import models.basicmodel
from util.lazy import lazy_scope_property
from tfcaps.layers import new_io


class ConvBaseline(models.basicmodel.BasicModel):
    
    @lazy_scope_property
    def logits(self):
        act = tf.nn.relu
        is_training = self.training
        
        i,o = new_io(self.img)

        i(tf.layers.conv2d(o(), filters=32, kernel_size=5, strides=1, padding='same', activation=act))
        i(tf.layers.max_pooling2d(o(), pool_size=2, strides=2))
        
        i(tf.layers.dropout(o(), rate=0.15, training=is_training))
        
        i(tf.layers.batch_normalization(o(), training=is_training))
        i(tf.layers.conv2d(o(), filters=64, kernel_size=3, strides=1, padding='same', activation=act))
        i(tf.layers.max_pooling2d(o(), pool_size=2, strides=2))
        
        i(tf.layers.dropout(o(), rate=0.15, training=is_training))
        
        i(tf.layers.batch_normalization(o(), training=is_training))
        i(tf.layers.conv2d(o(), filters=128, kernel_size=3, strides=1, padding='same', activation=act))
        i(tf.layers.max_pooling2d(o(), pool_size=2, strides=2))
        
        i(tf.layers.dropout(o(), rate=0.15, training=is_training))
        
        i(tf.layers.batch_normalization(o(), training=is_training))
        i(tf.layers.flatten(o()))
        i(tf.layers.dense(o(), 1024, activation=act))
        i(tf.layers.dropout(o(), rate=0.5, training=is_training))

        i(tf.layers.dense(o(), self.num_classes + self.garbage_class))
        
        return o()

    @lazy_scope_property
    def probabilities(self):
        return tf.nn.softmax(self.logits[:,:self.num_classes])
    
    @lazy_scope_property
    def prediction(self):
        return tf.argmax(self.probabilities, -1)

    @lazy_scope_property(only_training=True)
    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = opt.minimize(self.loss, global_step=tf.train.get_global_step())
        return train_op
    
    @lazy_scope_property(only_training=True)
    def summary(self):
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('Loss', self.loss)
        
    @lazy_scope_property
    def accuracy(self):
        correct_preds = tf.equal(self.prediction, self.label)
        return  tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.size(self.label), tf.float32)

    @lazy_scope_property
    def l2_loss(self):
        return self.l2_scale * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables(self.scope) if 'bias' not in v.name])
    

    @lazy_scope_property
    def loss(self):
        cross_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.label, self.num_classes+self.garbage_class), self.logits)
        return cross_loss + self.l2_loss
