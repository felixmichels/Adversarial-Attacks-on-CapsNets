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
from util.config import cfg
from util.util import load_cifar10, get_dir, get_model

def train_epoch(sess, model, init, writer):
    sess.run(init)
    try:
        while True:
            _, loss, acc, step, summary = sess.run(
                    [model.optimizer, model.loss, model.accuracy, tf.train.get_global_step(), summary_op],
                    feed_dict={model.train_placeholder: True})
                
            tf.logging.log_every_n(tf.logging.INFO,
                                   '| Steps: %5d | Loss: %5.1f | Accuracy: %1.3f',
                                    cfg.train_log_every_n,
                                    step, loss, acc)
            
            writer.add_summary(summary, step=tf.train.get_global_step())
            
    except tf.errors.OutOfRangeError:
        pass
    
def test(sess, model, init, writer):
    sess.run(init)
    acc_list = []
    try:
        while True:
            acc_list.append(sess.run(model.accuracy, feed_dict={model.train_placeholder: False}))
    except tf.errors.OutOfRangeError:
        pass
    acc = np.average(acc_list)
    tf.logging.info('\nTest Accuracy: %1.3f', np.average(acc))
    summary = tf.Summary(value=[
            tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    writer.add_summary(summary, step=tf.train.get_global_step())
    
def train_with_test(sess, model, train_init, test_init, ckpt_dir, log_dir):
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    
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
                
        if (ep+1)%cfg.save_every_n == 0 or time.time()-last_safe_time > cfg.save_freq:
            ckpt_path = os.path.join(ckpt_dir, 'model')
            saver.save(sess, ckpt_path, global_step=tf.train.get_global_step())
            last_safe_time = time.time()
            
        if (ep+1)%cfg.test_every_n == 0:
            test(sess, model, test_init, writer)
            
        sess.run(epoch_op)

    test(sess, model, test_init, writer)
    writer.close()


summary_op = None
epoch = None
epoch_op = None

def main(args):
    global summary_op, epoch, epoch_op
    
    if cfg.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
        
    model_name, model_class = get_model(args[1])   

    ckpt_dir = get_dir(cfg.ckpt_dir, model_name)
    log_dir = get_dir(cfg.log_dir, model_name)
    
    with tf.variable_scope('data'):
        tf.logging.debug('Load data')
        train_data = load_cifar10(is_train=True, batch_size=cfg.batch_size)
        test_data  = load_cifar10(is_train=False, batch_size=cfg.batch_size)
        
        iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                               train_data.output_shapes)
        img, label = iterator.get_next()
        
        tf.logging.debug('Creating iterator initializer')
        train_init = iterator.make_initializer(train_data)
        test_init = iterator.make_initializer(test_data)	  

    tf.train.create_global_step()
    epoch = tf.get_variable('epoch', dtype=tf.int32, initializer=0, trainable=False)
    tf.summary.scalar('Epoch', epoch)
    epoch_op = tf.assign_add(epoch, 1)
    summary_op = tf.summary.merge_all()
    
    tf.logging.debug('Creating model graph')
    model = model_class(img=img, label=label)
    
    tf.logging.debug('Creating summary op')
    summary_op = tf.summary.merge_all() 
    
    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            train_with_test(sess, model, train_init, test_init, ckpt_dir, log_dir)
        except KeyboardInterrupt:
            print("Manual interrupt occured")

if __name__ == '__main__':
    tf.app.run()