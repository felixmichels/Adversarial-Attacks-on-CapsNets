#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:44:07 2019

@author: felix
"""

import inspect
import tensorflow as tf
from util.lazy import lazy_scope_property

class BasicModel(object):
    
    def __init__(self, img, label, scope=None):
        self.img = img
        self.label = label
        
        if scope is not None:
            self.scope = scope
        else:
            self.scope = self.__class__.__name__
            
        with tf.variable_scope(self.scope):
            inspect.getmembers(self)
            
        
    @lazy_scope_property
    def train_placeholder(self):
        return tf.placeholder(dtype=tf.bool, shape=())