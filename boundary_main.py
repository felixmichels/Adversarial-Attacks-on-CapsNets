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
from multiprocessing.dummy import Pool
from util.config import cfg
from util.util import get_dir, get_model
from util.data import get_attack_original
from attacks.boundary_attack import boundary_attack


def is_adv_func(sess, model, true_label):
    def is_adv(img):
        acc = sess.run(model.accuracy,
                       feed_dicy = {
                               model.img: np.expand_dims(img, axis=0),
                               model.label: np.expand_dims(true_label, axis=0)
                               })
        return acc < 0.5
    return is_adv


def attack(sess, model, img, label):
    is_adv = is_adv_func(sess, model, label)
    return boundary_attack(img, is_adv, eps_min=cfg.eps_min, max_steps=cfg.max_steps)


def create_adv(sess, model):
    img, label = get_attack_original(cfg.attack_name, n=cfg.number_img)
    att_dir = get_dir(cfg.data_dir, cfg.attack_name, model.name)
    att_file = os.path.join(att_dir, 'adv_images.npy')
    if os.path.isfile(att_file):
        tf.logging.debug('Loading adv img from %s', att_file)
        adv_img = np.load(att_file)
    else:
        tf.logging.debug('Creating empty array for adv img')
        adv_img = np.empty(shape=(0,32,32,3), dtype='float32')
        
    num_adv = len(adv_img)
    adv_at_once = 2*cfg.processes
    tf.logging.info('Starting at img %d', num_adv)
    for i in range(num_adv, cfg.number_img, adv_at_once):
        tic = time.time()
        idx = slice(i, i+adv_at_once)
        with Pool(cfg.processes) as pool:
            adv_img[idx] = pool.starmap(
                lambda x,y: attack(sess, model, x, y),
                zip(img[slice], label[slice]))
        np.save(att_file, adv_img)
        tf.logging.info('Number of adv images: %d', i)
        tf.logging.info('Finished iteration in %.2f', time.time()-tic)
        
def main(args):
    
    if cfg.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
        
    model_class = get_model(args[1])
    
    tf.logging.debug('Creating model graph')
    img_ph = tf.placeholder(dtype=tf.float32, shape=(None,32,32,3))
    label_ph = tf.placeholder(dtype=tf.int64, shape=(None,))
    model = model_class(img=img_ph, label=label_ph, trainable=False)
    
    ckpt_dir = get_dir(cfg.ckpt_dir, model.name)
    saver = tf.train.Saver(var_list=tf.global_variables(scope=model.scope))
    
    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, save_path)
            create_adv(sess, model)
        except KeyboardInterrupt:
            print("Manual interrupt occured")
        

if __name__ == '__main__':
    tf.app.run()
