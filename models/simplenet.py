#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:36:43 2019

@author: felix
"""

import tensorflow as tf
import models.basicmodel
from util.lazy import lazy_scope_property

class SimpleNet(models.basicmodel.BasicModel):
         
    @lazy_scope_property
    def logits(self):
        w = tf.get_variable(name='weights', shape=(32*32*3, 10), initializer=tf.random_normal_initializer(0, 0.01))
        b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())
        return tf.matmul(tf.layers.flatten(self.img), w) + b
        
    @lazy_scope_property
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @lazy_scope_property(only_training=True)
    def optimizer(self):
        opt = tf.train.AdamOptimizer()
        return opt.minimize(self.loss, global_step=tf.train.get_global_step())
    
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
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.one_hot(self.label,10), name='entropy')
        return tf.reduce_mean(entropy, name='loss')