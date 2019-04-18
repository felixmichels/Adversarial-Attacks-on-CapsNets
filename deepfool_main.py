#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:02:25 2019

@author: felix
"""
import os
import time
import tensorflow as tf
import numpy as np
from util.config import cfg
from util.util import get_dir, get_model, np_save_bak
from util.data import get_attack_original
from attacks.deepfool import DeepfoolOp
from util.imgdataset import dataset_by_name


def create_adv(sess, dataset, deepfool):
    img, label = get_attack_original(cfg.attack_name, dataset, n=cfg.number_img)
    att_dir = get_dir(cfg.data_dir, dataset.name, cfg.attack_name, deepfool.model.name)

    att_file = os.path.join(att_dir, 'adv_images.npy')
    if os.path.isfile(att_file):
        tf.logging.debug('Loading adv img from %s', att_file)
        adv_img = np.load(att_file)
    else:
        tf.logging.debug('Creating empty array for adv img')
        adv_img = np.empty(shape=(0, *dataset.shape), dtype='float32')
        
    num_adv = len(adv_img)
    tf.logging.info('Starting at img %d', num_adv)
    for i in range(num_adv, cfg.number_img):
        tic = time.time()
        adv = deepfool.attack(sess, img[i], label[i], cfg.max_iter, cfg.step_size)

        if adv is None:
            tf.logging.info('Attack failed...')
            # Try again with bigger step size
            adv = deepfool.attack(sess, img[i], label[i], cfg.max_iter, cfg.max_step_size)
        if adv is None:
            tf.logging.info('Failed a second time...')
            # If attack didn't succeed, mark image with NaN
            adv = np.empty_like(img[i])
            adv[:] = np.nan

        adv_img = np.append(adv_img, [adv], axis=0)
        np_save_bak(att_file, adv_img)
        tf.logging.info('Number of adv images: %d', i)
        tf.logging.info('Finished iteration in %.2f', time.time()-tic)


def main(args):
    
    model_class = get_model(args[1])
    dataset = dataset_by_name(args[2])

    params = get_param(args[1], dataset.name)
    
    tf.logging.debug('Creating attack ops')
    deepfool = DeepfoolOp(model_class,
                          dataset=dataset
                          params=params)
    
    ckpt_dir = get_dir(cfg.ckpt_dir, dataset.name, deepfool.model.name)
    saver = tf.train.Saver(var_list=tf.global_variables(scope=deepfool.model.scope))
    
    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            sess.graph.finalize()
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, save_path)
            create_adv(sess, dataset, deepfool)
        except KeyboardInterrupt:
            print("Manual interrupt occured")
        

if __name__ == '__main__':
    tf.app.run()
