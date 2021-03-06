#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training and testing architectures in the model folder
"""

import os
import time
import tensorflow as tf
import numpy as np
from util.config import cfg, load_config
from util.util import get_dir, get_model, create_epoch, get_epoch, get_epoch_op, get_params
from util.data import to_tf_dataset
from util.imgdataset import dataset_by_name


def train_epoch(sess, model, init, writer=None):
    """Trains a single epoch and writes to summary (but doesn't save)"""
    sess.run(init)
    acc_list = []
    try:
        while True:
            _, loss, acc, step, summary = sess.run(
                [
                    model.optimizer,
                    model.loss,
                    model.accuracy,
                    tf.train.get_global_step(),
                    model.summary_op
                ],
                feed_dict={model.training: True})

            acc_list.append(acc)
            tf.logging.log_every_n(
                tf.logging.INFO,
                '| Steps: %5d | Loss: %5.3f | Accuracy: %1.3f',
                cfg.train_log_every_n,
                step, loss, acc)

            if step % cfg.summary_every_n == 0:
                tf.logging.debug('Writing summary')
                if writer is not None:
                    writer.add_summary(summary, global_step=step)

    except tf.errors.OutOfRangeError:
        pass
    acc = np.average(acc_list)
    tf.logging.info('Average accuracy this epoch: %1.3f', acc)


def test(sess, model, init, writer=None):
    """Runs the test set and averages accuracy"""
    sess.run(init)
    acc_list = []
    try:
        while True:
            acc_val = sess.run(model.accuracy,
                               feed_dict={model.training: False})
            acc_list.append(acc_val)
    except tf.errors.OutOfRangeError:
        pass
    acc = np.average(acc_list)
    tf.logging.info('\nTest Accuracy: %1.3f', np.average(acc))
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=model.scope+"/test_accuracy", simple_value=acc)])
    step = sess.run(tf.train.get_global_step())
    if writer is not None:
        tf.logging.debug('Writing test summary')
        writer.add_summary(summary, global_step=step)


def train_with_test(sess, model, train_init, test_init, ckpt_dir, log_dir):
    """Loads the model (if --reuse) and trains remaining epochs"""
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=2)
    if cfg.no_summary:
        writer = None
    else:
        writer = tf.summary.FileWriter(log_dir, sess.graph)

    if cfg.restore:
        save_path = tf.train.latest_checkpoint(ckpt_dir)
        saver.restore(sess, save_path)

    last_safe_time = time.time()
    while sess.run(get_epoch()) < cfg.epochs:
        ep_start_time = time.time()

        ep = sess.run(get_epoch())
        tf.logging.info('Epoch: %d', ep)
        train_epoch(sess, model, train_init, writer)
        tf.logging.info('Time: %5.2f', time.time()-ep_start_time)

        if (ep+1) % cfg.test_every_n == 0:
            test(sess, model, test_init, writer)

        sess.run(get_epoch_op())

        if ((ep+1) % cfg.save_every_n == 0 or time.time()-last_safe_time > cfg.save_freq) and not cfg.no_save:
            ckpt_path = os.path.join(ckpt_dir, 'model')
            saver.save(sess, ckpt_path, global_step=tf.train.get_global_step())
            last_safe_time = time.time()
            tf.logging.info('Saved to %s', ckpt_dir)

    test(sess, model, test_init, writer)
    if writer is not None:
        writer.close()


def main(args):

    model_class = get_model(args[1])
    dataset = dataset_by_name(args[2])

    params = get_params(args[1], dataset.name)
    load_config(dataset.name, optional=True)

    with tf.variable_scope('data'):
        tf.logging.debug('Load data')
        train_data = to_tf_dataset(dataset, is_train=True, batch_size=cfg.batch_size,
                                   aug_strength=cfg.data_aug, 
                                   aug_prob=cfg.aug_prob,
                                   aug_flip=cfg.aug_flip)
        test_data = to_tf_dataset(dataset, is_train=False, batch_size=cfg.batch_size)

        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
        img, label = iterator.get_next()

        tf.logging.debug('Creating iterator initializer')
        train_init = iterator.make_initializer(train_data)
        test_init = iterator.make_initializer(test_data)

    tf.train.create_global_step()
    create_epoch()

    tf.logging.debug('Creating model graph')
    model = model_class(img=img, label=label, num_classes=dataset.num_classes, **params)

    ckpt_dir = get_dir(cfg.ckpt_dir, dataset.name, model.name)
    log_dir = get_dir(cfg.log_dir, dataset.name, model.name)

    if cfg.stop_before_session:
        exit()

    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            train_with_test(sess, model, train_init, test_init, ckpt_dir, log_dir)
        except KeyboardInterrupt:
            print("Manual interrupt occured")


if __name__ == '__main__':
    tf.app.run()
