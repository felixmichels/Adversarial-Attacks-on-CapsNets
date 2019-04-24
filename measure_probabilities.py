#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:28:14 2019

@author: felix
"""

import os
import numpy as np
import tensorflow as tf
from util.util import get_model, get_dir, get_params
from util.config import cfg
from util.imgdataset import dataset_by_name


def get_probabilities(model, img, session, batch_size):
    probs = None
    N = len(img)
    for idx in np.array_split(np.arange(N), N//batch_size):
        batch_probs = session.run(
            model.probabilities,
            feed_dict={model.img: img[idx]})
        if probs is None:
            probs = batch_probs
        else:
            probs = np.concatenate([probs, batch_probs])
    return probs


def measure_normal_attack(attack_name, dataset_name, model, sess, source_name):
    tf.logging.info('Measuring %s: source %s, target %s', attack_name, source_name, model.name)

    source_path = os.path.join(cfg.data_dir, dataset_name, attack_name, source_name)
    adv_files = [fn for fn in os.listdir(source_path) if fn.endswith('.npy')]

    assert len(adv_files) == 1, 'No valid .npy file found for {},{}'.format(attack_name, source_name)

    adv_file_name = adv_files[0]
    adv_file = os.path.join(source_path, adv_file_name)

    adv_img = np.load(adv_file)
    tf.logging.info('Loaded adv images %s', adv_file)

    probs = get_probabilities(model, adv_img, sess, cfg.batch_size)

    save_dir = get_dir(cfg.data_dir, dataset_name, attack_name, 'Measure_' + source_name + '_' + model.name)
    save_file = os.path.join(save_dir, 'probabilities_' + adv_file_name)

    np.save(save_file, probs)


def measure_universal(dataset_name, model, sess, source_name):
    attack_name = 'universal_perturbation'
    tf.logging.info('Measuring %s: source %s, target %s', attack_name, source_name, model.name)

    attack_path = os.path.join(cfg.data_dir, dataset_name, attack_name, 'originals.npz')
    with np.load(attack_path) as npz:
        img = npz['img']

    source_path = os.path.join(cfg.data_dir, dataset_name, attack_name, source_name)
    adv_files = [fn for fn in os.listdir(source_path) if fn.endswith('.npy')]
    for adv_file_name in adv_files:
        pert_file = os.path.join(source_path, adv_file_name)
        perts = np.load(pert_file)
        tf.logging.info('Loaded adv images %s', pert_file)

        prob_list = []
        for pert in perts:
            prob_list.append(get_probabilities(model, img+pert, sess, cfg.batch_size))

        save_dir = get_dir(cfg.data_dir, dataset_name, attack_name, 'Measure_' + source_name + '_' + model.name)
        save_file = os.path.join(save_dir, 'probabilities_' + adv_file_name)

        np.save(save_file, prob_list)


def measure__attack(attack_name, dataset_name, model, sess, source_name):
    if attack_name != 'universal_perturbation':
        measure_normal_attack(attack, dataset_name, model, sess, source_name)
    else:
	measure_universal(dataset_name, model, sess, source_name)


def main(args):

    model_name = args[1]
    model_class = get_model(model_name)
    source_name = args[2]
    dataset = dataset_by_name(args[3])

    if len(args < 5):
        attacks = ['carlini_wagner', 'boundary_attack', 'deepfool', 'universal_perturbation']
    else:
        attacks = args[4].split(',')

    params = get_params(model_name, dataset.name)

    tf.logging.debug('Creating model graph')
    model = model_class(trainable=False, num_classes=dataset.num_classes, shape=dataset.shape, **params)

    ckpt_dir = get_dir(cfg.ckpt_dir, dataset.name, model.name)
    saver = tf.train.Saver(var_list=tf.global_variables(scope=model.scope))

    if cfg.stop_before_session:
        exit()

    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, save_path)
	    for attack in attacks:
                measure_attack(attack, dataset.name, model, sess, source_name)

        except KeyboardInterrupt:
            print("Manual interrupt occurred")


if __name__ == '__main__':
    tf.app.run()
