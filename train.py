#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:06:52 2019

@author: felix
"""

import tensorflow as tf
import numpy as np
import os
import time
import importlib
import inspect
from util.config import cfg
from util.util import load_cifar10

def train_epoch(sess, model, init):
    sess.run(init)
    try:
        while True:
                _, loss, acc, step = sess.run(
                        [model.optimizer, model.loss, model.accuracy, tf.train.get_global_step()],
                        feed_dict={model.training_placeholder: True})
                
                tf.logging.log_every_n(20,
                                       '| Steps: %5d | Loss: %5.1f | Accuracy: %1.3f',
                                       50,
                                       step, loss, acc)
    except tf.errors.OutOfRangeError:
        pass
    
def test(sess, model, init):
    sess.run(init)
    acc_list = []
    try:
        while True:
            acc_list.append(sess.run(model.accuracy, feed_dict={model.training_placeholder: False}))
    except tf.errors.OutOfRangeError:
        pass
    tf.logging.info('\nTest Accuracy: %1.3f', np.average(acc_list))
    
def train_with_test(sess, model, train_init, test_init, ckpt_dir):
    epoch = tf.get_variable('epoch', dtype=tf.int32, initializer=0, trainable=False)
    epoch_op = tf.assign_add(epoch, 1)
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
    
    if cfg.restore:
        save_path = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, save_path)

    last_safe_time = float('inf')
    while sess.run(epoch) < cfg.epochs:
        ep_start_time = time.time()
        
        ep = sess.run(epoch)
        tf.logging.info('Epoch: %d', ep)
        train_epoch(sess, model, train_init)
        tf.logging.info('Time: %5.2f', time.time()-ep_start_time)
                
        if ep > 0 and (ep%5 == 0 or time.time()-last_safe_time > 300):
            saver.save(sess, ckpt_dir, global_step=tf.train.get_global_step())
            last_safe_time = time.time()
            
        if ep > 0 and ep%5 == 0:
            test(sess, ckpt_dir, model, test_init)
            
        sess.run(epoch_op)

def main(args):
    model_name = args[1].lower()
    # Evil reflection
    model_module = importlib.import_module('.'+model_name,cfg.model_pck)
    [(model_name, model_class)] = inspect.getmembers(model_module, inspect.isclass)
    
    ckpt_dir = os.join(cfg.ckpt_dir, model_name)
    
    train_data = load_cifar10(is_train=True, batch_size=cfg.batch_size)
    test_data  = load_cifar10(is_train=False, batch_size=cfg.batch_size)
    
    iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
    img, label = iterator.get_next()
    
    train_init = iterator.make_initializer(train_data)
    test_init = iterator.make_initializer(test_data)	  

    tf.train.create_global_step()
    model = model_class(img=img, label=label)
    
    with tf.Session() as sess:
        try:
            train_with_test(sess, model, train_init, test_init, ckpt_dir)
        except KeyboardInterrupt:
            print("Manual interrupt occured")

if __name__ == '__main__':
    tf.app.run()