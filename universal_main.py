#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:46:43 2019

@author: felix
"""

import os
import time
import tensorflow as tf
import numpy as np
from util.config import cfg
from util.util import get_dir, get_model, np_save_bak
from util.data import get_attack_original
from attacks.universal_perturbation import UniversalPerturbation
from attacks.fast_batch_attacks import FGSM

fgsm_eps = 0.05
max_it = 50
max_norm = 3.0 # Apprx 0.1 times average cifar10 image norm
batch_size=128
pert_per_split = 10
num_split = 10
attack_name='universal_perturbation'

def create_adv(sess, fgsm):
    img, label = get_attack_original(attack_name)
    att_dir = get_dir(cfg.data_dir, attack_name, fgsm.model.name)
    
    for idx in np.array_split(np.arange(len(label)), num_split):
        tf.logging.info('Attacking range {}-{}'.format(idx.min(), idx.max()))
        att_file = os.path.join(att_dir, 'adv_perts{}-{}.npy'.format(idx.min(), idx.max()))
        if os.path.isfile(att_file):
            tf.logging.debug('Loading pert from %s', att_file)
            perts = np.load(att_file)
        else:
            tf.logging.debug('Creating empty array for perts')
            perts = np.empty(shape=(0,32,32,3), dtype='float32')
            
        num_adv = len(perts)
        for i in range(num_adv, pert_per_split):
            attack = UniversalPerturbation(fgsm,
                                           img[idx], label[idx],
                                           batch_size=batch_size,
                                           max_it=max_it,
                                           max_norm=max_norm)
            tic = time.time()
            attack.fit(sess)
            perts = np.append(perts, [attack.perturbation], axis=0)
            np_save_bak(att_file, perts, num_bak=2)
            tf.logging.info('Finished iteration in %.2f', time.time()-tic)

        
def main(args):
    
    model_class = get_model(args[1])
    
    tf.logging.debug('Building model')
    model = model_class(tf.placeholder(dtype=tf.float32, shape=(None,32,32,3)),
                        tf.placeholder(dtype=tf.int64, shape=(None,)),
                        trainable=False)
    
    tf.logging.debug('Creating attack ops')
    fgsm = FGSM(model, fgsm_eps)
    
    ckpt_dir = get_dir(cfg.ckpt_dir, model.name)
    saver = tf.train.Saver(var_list=tf.global_variables(scope=model.scope))
    
    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, save_path)
            create_adv(sess, fgsm)
        except KeyboardInterrupt:
            print("Manual interrupt occured")
        

if __name__ == '__main__':
    tf.app.run()