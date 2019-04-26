#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:57:45 2019

@author: felix
"""

import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_string('attack_name', 'carlini_wagner', 'Name of the attack (used for directory name)')
flags.DEFINE_integer('number_img', 1000, 'Number of images to attack')

flags.DEFINE_integer('max_opt_iter', 10000, 'Maximum number of stepf for adam optimizer')
flags.DEFINE_integer('max_bin_iter', 5, 'Maximum number of steps in binary search')

flags.DEFINE_float('rand_start_std', 0.0, 'Standard deviation of random start image')
flags.DEFINE_float('kappa', 1.0, 'The confidence value kappa')
