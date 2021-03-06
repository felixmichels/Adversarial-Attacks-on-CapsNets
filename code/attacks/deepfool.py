#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFool
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
            grads.append(tf.gradients(y[k], x)[0])
        return tf.stack(grads, axis=0)


class DeepfoolOp():
    def __init__(self,
                 model_class,
                 dataset,
                 params):
        """
        Creates necessary ops for deepfool in the tensorflow graph
        Args:
            model_class: The class of the model to construct.
                         Expects subclass of BasicModel
            dataset: The dataset to use.
                     Only necessary for shape and number of classes
            params: Additional parameters to pass to the model init
        """
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
        """
        Deepfool attack
        Args:
            sess: Tensorflow session
            original: original image to attack
            label: true label of original
            max_iter: maximum number of iterations
            step_size: Limit each step to this value


        Returns: Returns a valid adversarial example,
                 or None if max_iter was reached
        """
        adv = original.copy()
        dtype = np.float32
        # For numerical stability
        eps = 1e-5

        ks = [k for k in range(self.num_classes) if k != label]
        w = np.empty(shape=(self.num_classes,)+original.shape, dtype=dtype)
        f = np.empty(shape=self.num_classes, dtype=dtype)
        l = np.empty(shape=self.num_classes, dtype=dtype)
        l[label] = np.inf
        
        for it in range(max_iter):
            
            logits, logit_grads = sess.run(
                    [self.logits, self.logits_grad],
                    feed_dict={self.image: adv})
            
            for k in ks:
                w[k] = logit_grads[k] - logit_grads[label]
                f[k] = logits[k] - logits[label]
                l[k] = np.abs(f[k]) / la.norm(w[k])

            if (f[ks] > eps).any():
                return adv
                
            l_hat = np.argmin(l)

            pert = ((np.abs(f[l_hat]) + eps) / la.norm(w[l_hat])**2) * w[l_hat] 
            pert *= min(1.0, step_size/(la.norm(pert)+1e-8))
            adv = np.clip(adv+pert, 0, 1)
