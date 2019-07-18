#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carlini-Wagner attack
"""

import tensorflow as tf


class CWAttackOp():
    def __init__(self,
                 model_class,
                 model_params,
                 shape,
                 num_classes,
                 kappa,
                 rand_start_std=0.0):
        """
        Builds a optimize op for the carlini wagner attack.

        Args:
            model_class: The class of the model to construct.
                         Expects subclass of BasicModel
            model_params: Additional parameter for the model init
            shape: Shape of the images
            num_classes: Number of classes
            kappa: Confidence parameter
            rand_start_std: std if we want to start from a
                            randomly disturbed image
        """

        self.original = tf.placeholder(dtype=tf.float32, shape=shape)
        self.target = tf.placeholder(dtype=tf.int64, shape=())
        self.lagrangian = tf.placeholder(dtype=tf.float32, shape=())

        initial_im = tf.clip_by_value(
            self.original + tf.random_normal(self.original.get_shape(),
                                             mean=0.0, stddev=rand_start_std),
            clip_value_min=0.001,
            clip_value_max=0.999)
        w = tf.get_variable('w', trainable=True,
                            initializer=tf.atanh(2*initial_im - 1))

        image = (0.5 + 1e-8) * (tf.tanh(w) + 1)

        self.model = model_class(
            tf.expand_dims(image, axis=0),
            tf.expand_dims(self.target, axis=0),
            num_classes=num_classes,
            trainable=False,
            **model_params)

        logits = tf.squeeze(self.model.logits)
        mask = tf.logical_not(
                tf.cast(tf.one_hot(self.target, num_classes), tf.bool))

        target_logits = logits[self.target]
        others_logits = tf.reduce_max(tf.boolean_mask(logits, mask))

        # add kappa, so that adv_loss >= 0
        adv_loss = kappa + tf.maximum(others_logits - target_logits, -kappa)
        pert_norm = tf.norm(image-self.original)
        loss = pert_norm + self.lagrangian*adv_loss
        opt = tf.train.AdamOptimizer()
        opt_op = opt.minimize(
            loss,
            var_list=[w],
            global_step=tf.train.get_or_create_global_step())

        init = tf.variables_initializer(
                opt.variables() + [w, tf.train.get_global_step()])
        target_reached = self.model.accuracy > 0.5
        
        self.loss = loss
        self.target_reached = target_reached
        self.optimizer = opt_op
        self.init = init
        self.image = image


def cw_attack(sess, orig, target, attack, max_opt_iter, max_bin_iter, c_prop):
    """
    Carlini wagner attack for a single image
    
    Args:
        sess: The tensorflow session to use
        orig: The original image (as an numpy array) to attack
        target: The target label
        attack: An instance of CWAttackOp
        max_opt_iter: Maximal number of steps for the adam optimizer
        max_bin_iter: Maximal number of steps for the binary search for c
        c_prop: A property for the initial c value.
                Is set to the value returned by binary search

    Returns:
        An numpy array containing the adversarial example,
        or None if the attack was unsuccessful
    """
      
    def test_func(c):
        tf.logging.info('Testing adv img with c=%f', c)
        sess.run(attack.init, feed_dict={attack.original: orig})
        loss_prev = 1e6
        for it in range(max_opt_iter):
            _, loss = sess.run(
                [attack.optimizer, attack.loss],
                feed_dict = {
                    attack.original: orig,
                    attack.target: target,
                    attack.lagrangian: c,
                })
            if it % (max_opt_iter // 10) == 0:
                tf.logging.debug('Loss: %f', loss)
                # Check if improvement is made
                if loss > 0.999 * loss_prev:
                    tf.logging.debug('Stopped early')
                    break
                loss_prev = loss
                
        return sess.run(attack.target_reached,
                        feed_dict = {attack.target: target})
        
    tf.logging.debug('Starting binary search')
    c = _binary_search_min(test_func, init_c=c_prop.fget(),
                           max_iter=max_bin_iter)
    if c is None:
        return None

    c_prop.fset(c)
    
    sess.run(attack.init, feed_dict={attack.original: orig})
    good_adv = None
    tf.logging.info('Final run with c=%f', c)
    for it in range(max_opt_iter):
        _ = sess.run(
            attack.optimizer,
            feed_dict={
                attack.original: orig,
                attack.target: target,
                attack.lagrangian: c,
            })
        if it % (max_opt_iter // 10) == 0 or it == max_opt_iter-1:
            success, adv = sess.run([attack.target_reached, attack.image],
                                    feed_dict={attack.target: target})
            if success:
                good_adv = adv
            elif good_adv is not None:
                tf.logging.debug('Early stopping in final run')
                break
    return good_adv


def _binary_search_min(func, init_c, max_iter=10):
    """Finds minimal c, s.t. func(c) is True
    Assumes, that func is monotone and func(0) is False
    Returns None, if no valid c could be found"""
    c_min = 0.0
    c_max = None
    c = init_c

    for _ in range(max_iter):
        if func(c):
            c_max = c
        else:
            c_min = c

        if c_max is None:
            c *= 2
        else:
            c = (c_max + c_min) / 2

    # Final test
    if not func(c):
        c = c_max

    # Return slightly larger c for safety
    if c_max is not None:
        c += 1e-3*(c_max - c_min)
        return c + 1e-3*(c_max - c_min)
