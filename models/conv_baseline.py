#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:35:34 2019

@author: felix
"""

import tensorflow as tf
import models.basicmodel
from util.lazy import lazy_scope_property
from util.config import cfg

class ConvBaseline(models.basicmodel.BasicModel):
    
    @lazy_scope_property
    def logits(self):
        act = tf.nn.relu
        is_training = self.train_placeholder
        
        
        conv1 = tf.layers.conv2d(self.img, filters=32, kernel_size=5, strides=1, padding='same', activation=act)
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=5, strides=1, padding='same', activation=act)
        pool1 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
        bn1 = tf.layers.batch_normalization(pool1, training=is_training)
        
        drop1 = tf.layers.dropout(bn1, rate=0.15, training=is_training)
        
        conv3 = tf.layers.conv2d(drop1, filters=64, kernel_size=3, strides=1, padding='same', activation=act)
        conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=3, strides=1, padding='same', activation=act)
        pool2 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)
        bn2 = tf.layers.batch_normalization(pool2, training=is_training)
        
        drop2 = tf.layers.dropout(bn2, rate=0.2, training=is_training)
        
        conv5 = tf.layers.conv2d(drop2, filters=128, kernel_size=3, strides=1, padding='same', activation=act)
        conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=3, strides=1, padding='same', activation=act)
        pool3 = tf.layers.max_pooling2d(conv6, pool_size=2, strides=2)
        bn3 = tf.layers.batch_normalization(pool3, training=is_training)
        
        drop3 = tf.layers.dropout(bn3, rate=0.15, training=is_training)
        
        flat = tf.layers.flatten(drop3)
        dense1 = tf.layers.dense(flat, 1024, activation=act)
        drop4 = tf.layers.dropout(dense1, rate=0.5, training=is_training)
        
        dense2 = tf.layers.dense(drop4, cfg.classes)
        
        return dense2
    
    @lazy_scope_property
    def prediction(self):
        return tf.nn.softmax(self.logits)

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
        correct_preds = tf.equal(tf.argmax(self.prediction, 1), self.label)
        return  tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / tf.cast(tf.size(self.label), tf.float32)

    @lazy_scope_property
    def loss(self):
        cross_loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.label, cfg.classes), self.logits)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss = cross_loss + 1e-5 * l2_loss
        return loss