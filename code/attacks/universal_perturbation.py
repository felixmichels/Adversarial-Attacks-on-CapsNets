#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:08:06 2019

@author: felix
"""

import numpy as np
import tensorflow as tf
import numpy.linalg as la

        
def _clip(img, norm):
    np.clip(img, -1, 1, out=img)
    if norm is not None and la.norm(img) >= norm:
        img *= norm / la.norm(img)

class UniversalPerturbation():
    
    def __init__(self,
                 attack,
                 img, label,
                 batch_size,
                 max_it,
                 max_norm=None,
                 target_rate=None):
        """

        Args:
            attack: FastAttack instance
            img: numpy array of originals
            label: original labels
            batch_size: used in FastAttack
            max_it: Maximal number of iterations
            max_norm: Maximal l2-norm of perturbation,
                      or None for no restriction
            target_rate: Stop, if this fooling rate is reached
        """

        self.target_rate = target_rate
        self.attack = attack
        self.img = img
        self.label = label
        self.batch_size = batch_size
        self.max_it = max_it
        self.max_norm = max_norm

        self.perturbation = np.zeros_like(img[0])
        self._best_fool_rate = 1.0
        self._work_pert = None
        
    def _feed(self, idx):
        return {self.attack.model.img: np.clip(self.img[idx] + self._work_pert, 0, 1),
                self.attack.model.label: self.label[idx]}
            
    def _not_fooled(self, sess):
        preds = []
        N = len(self.label)
        for idx in np.array_split(np.arange(N), N//self.batch_size):
            preds.append(sess.run(
                    self.attack.model.prediction,
                    feed_dict=self._feed(idx)))
            
        preds = np.concatenate(preds)
        return preds == self.label
            
    def fit(self, sess):
        self._work_pert = np.random.normal(size=self.perturbation.shape).astype('float32')
        if self.max_norm is None:
            init_norm = np.mean([la.norm(img) for img in self.img]) / 100
        else:
            init_norm = self.max_norm / 10

        _clip(self._work_pert, init_norm)
        
        correct = self._not_fooled(sess)
        for it in range(self.max_it):
            
            idx = np.random.choice(np.where(correct)[0], self.batch_size)
            pert = sess.run(self.attack.perturbation, feed_dict=self._feed(idx))
            pert = pert.mean(axis=0)
            self._work_pert += pert
                      
            _clip(self._work_pert, self.max_norm)
                
            correct = self._not_fooled(sess)
            acc = np.mean(correct)
            if acc < self._best_fool_rate:
                tf.logging.debug('Found good perturbation')
                self._best_fool_rate = acc
                self.perturbation = self._work_pert

            if self.target_rate is not None and self._best_fool_rate < self.target_rate:
                break
            
            tf.logging.debug('it: %d, acc: %1.3f, norm: %1.3f', it, acc, la.norm(self._work_pert))
            
        tf.logging.info('Reached acc %1.3f with norm %2.3f', self._best_fool_rate, la.norm(self.perturbation))
