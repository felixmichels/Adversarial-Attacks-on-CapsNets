#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:55:18 2019

@author: felix
"""

import tensorflow as tf
import numpy as np
import numpy.linalg as la

try:
    from tensorflow.python.ops.parallel_for.gradients import jacobian
    tf.app.flags.FLAGS.op_conversion_fallback_to_while_loop = True
except ImportError:
    tf.logging.info('Using own jacobian function (slow)')
    def jacobian(y,x):
        grads = []
        for k in range(y.shape.as_list()[0]):
            grads.append(tf.gradients(y[k],x)[0])
        return tf.stack(grads, axis=0)


class DeepfoolOp():
    def __init__(self,
                 model_class,
                 dataset,
                 params
                 ):
        self.image = tf.placeholder(dtype=tf.float32, shape=dataset.shape)
        
        self.model = model_class(
            tf.expand_dims(self.image, axis=0),
            trainable=False,
            num_classes=dataset.num_classes,
            **params)
        
        self.logits = self.model.logits[0]
        self.num_classes = self.logits.shape.as_list()[0]
        self.logits_grad = jacobian(self.logits, self.image)
        
    
    def attack(self,
               sess,
               original,
               label,
               max_iter,
               step_size):
        """Deepfool attack
        args:
            sess: Tensorflow session
            original: original image to attack
            label: true label of original
            attack: instance of 
           ...
        """
        adv = original.copy()

        ks = [k for k in range(self.num_classes) if k != label]
        w = np.empty(shape=(self.num_classes,)+original.shape, dtype='float32')
        f = np.empty(shape=self.num_classes, dtype='float32')
        l = np.empty(shape=self.num_classes, dtype='float32')
        l[label] = np.inf
        
        for it in range(max_iter):
            
            logits, logit_grads = sess.run(
                    [self.logits, self.logits_grad],
                    feed_dict={self.image: adv})
            
            for k in ks:
                w[k] = logit_grads[k] - logit_grads[label]
                f[k] = logits[k] - logits[label]
                l[k] = np.abs(f[k]) / la.norm(w[k])

            if (f[ks] > 1e-5).any():
                return adv
                
            l_hat = np.argmin(l)

            pert = ((np.abs(f[l_hat]) + 1e-5) / la.norm(w[l_hat])**2) * w[l_hat] 
            pert *= min(1.0, step_size/(la.norm(pert)+1e-8))
            adv = np.clip(adv+pert, 0, 1)
