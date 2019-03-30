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
        
        self.__initialize_properties()
        
        
    @property
    def name(self):
        return self.scope
        
    def __initialize_properties(self):
        # Set is_build = True, so that properties get added to respective lists
        self.is_build = True
        self.normal_vars = []
        self.training_vars = []
        
        #Fill normal_vars/training_vars
        inspect.getmembers(self)
        
        # Set is_buidl = False, so that properties have normal behaviour
        self.is_build = False
        #Initialize properties
        with tf.variable_scope(self.scope):
            for v in self.normal_vars:
                getattr(self,v)
                
            if self.trainable:
                for v in self.training_vars:
                    getattr(self,v)
                    
        delattr(self, 'normal_vars')
        delattr(self, 'training_vars')

            
        
    @lazy_scope_property
    def train_placeholder(self):
        if self.trainable:
            return tf.placeholder_with_default(True, shape=())
        else:
            return tf.constant(False)