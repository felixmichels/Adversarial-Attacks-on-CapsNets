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
    
    def __init__(self, img, label, trainable=True, scope=None):
        self.img = img
        self.label = label
        self.trainable = trainable

        self.scope = scope or self.__class__.__name__
        
        #Warning: Bad code ahead
        
        self.is_build = True
        self.normal_vars = []
        self.training_vars = []
        
        #Fill normal_vars/training_vars
        inspect.getmembers(self)
        
        #Initialize properties
        self.is_build = False
        with tf.variable_scope(self.scope):
            for v in self.normal_vars:
                getattr(self,v)
                
            if self.trainable:
                for v in self.training_vars:
                    getattr(self,v)

            
        
    @lazy_scope_property
    def train_placeholder(self):
        if self.trainable:
            return tf.placeholder_with_default(True, shape=())
        else:
            return tf.constant(False)