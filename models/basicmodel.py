#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:44:07 2019

@author: felix
"""

import inspect
import tensorflow as tf
from abc import ABC, abstractmethod
from util.lazy import lazy_scope_property


class BasicModel(ABC):
    """
    Base class for model objects.
    Designed to work with lazy_scope_property;
    thus decorated methods are initialized immediately and only once,
    in an guaranteed order
    """
    
    def __init__(self,
                 img=None,
                 label=None,
                 num_classes=10,
                 trainable=True,
                 scope=None,
                 shape=(32, 32, 3),
                 **kwargs):
        """
        img: Tensor, input images. If None, uses placeholder
        label: Tensor, labels. If None, uses placeholder
        trainable: Boolean, indicates if model is trainable
        scope: Optional scope name for tensors in this model.
               Uses the class name (of the subclass) as default
        shape: Input shape. Only used, if img is None
        kwargs: Optional parameters
        """
        self.img = img if img is not None else tf.placeholder(dtype=tf.float32, shape=(None, *shape))
        self.label = label if label is not None else tf.placeholder(dtype=tf.int64, shape=(None,))
        self.trainable = trainable
        self.num_classes = num_classes
        self.garbage_class = 0

        self.scope = scope or self.__class__.__name__

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__initialize_properties()

    def __initialize_properties(self):
        # Set is_build = True, so that properties get added to respective lists
        self.is_build = True
        self.normal_vars = []
        self.training_vars = []

        # Fill normal_vars/training_vars
        inspect.getmembers(self)

        self.normal_vars = sorted(self.normal_vars,
                                  key=lambda s: '' if s == 'probabilities' else s)
        self.training_vars = sorted(self.training_vars)

        tf.logging.debug('%s properties: %s', self.name, self.normal_vars)
        tf.logging.debug('%s training: %s', self.name, self.normal_vars)

        # Set is_build = False, so that properties have normal behaviour
        self.is_build = False
        # Initialize properties
        with tf.variable_scope(self.scope):
            for v in self.normal_vars:
                getattr(self, v)

            if self.trainable:
                for v in self.training_vars:
                    getattr(self, v)

        # Var lists are not needed anymore,
        # but MUST keep self.is_build for correct lazy_scope_property behaiour
        delattr(self, 'normal_vars')
        delattr(self, 'training_vars')

    @property
    def name(self):
        """
        The name of the model.
        May depend on hyperparameters and should be used
        for save path names etc.
        """
        return self.scope
    
    @lazy_scope_property(only_training=True)
    def summary_op(self):
        """Summary op for this model"""
        return tf.summary.merge_all(scope=self.scope)
        
    @lazy_scope_property
    def training(self):
        """
        Training placeholder for use with tf session feed_dict.
        If the model is not trainable, returns a tf constant False
        """
        if self.trainable:
            return tf.placeholder_with_default(False, shape=())
        return tf.constant(False)

    @abstractmethod
    def loss(self):
        """Total loss"""
        pass
    
    @abstractmethod
    def accuracy(self):
        """Accuracy for this batch"""
        pass
    
    @abstractmethod
    def optimizer(self):
        """Optimizer op"""
        pass
