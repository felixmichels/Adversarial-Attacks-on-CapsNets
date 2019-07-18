#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attacks for whole batches.
Can be used in the universal perturbations
"""

import tensorflow as tf
from abc import ABC, abstractmethod
from util.lazy import lazy_property


class FastAttack(ABC):
    
    def __init__(self, model):
        self.model = model
        self.perturbation
        
    @lazy_property
    def perturbation(self):
        return tf.clip_by_value(self._unclipped_pert(),
                                0 - self.model.img,
                                1 - self.model.img)
    
    @abstractmethod
    def _unclipped_pert(self):
        pass
    
    
class FGSM(FastAttack):
    
    def __init__(self, model, epsilon):
        self.epsilon = epsilon
        super(self.__class__, self).__init__(model)
        
    def _unclipped_pert(self):
        grad = tf.gradients(self.model.loss, self.model.img)[0]
        return self.epsilon*tf.sign(grad)