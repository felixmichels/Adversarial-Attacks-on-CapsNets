#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:44:07 2019

@author: felix
"""

import abc

class BasicModel(object, metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def probabilities(self):
         raise NotImplementedError()
         
    @property
    @abc.abstractmethod
    def logits(self):
        raise NotImplementedError()
        
    @property
    @abc.abstractmethod
    def prediction(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def optimizer(self):
        raise NotImplementedError()
        
    @property
    @abc.abstractmethod
    def accuracy(self):
        raise NotImplementedError()
        
    @property
    @abc.abstractmethod
    def loss(self):
        raise NotImplementedError()
        
    @property
    @abc.abstractmethod
    def train_placeholder(self):
        raise NotImplementedError()
        
    @property
    @abc.abstractmethod
    def scope(self):
        raise NotImplementedError()