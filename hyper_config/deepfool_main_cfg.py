#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:57:45 2019

@author: felix
"""

import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_string('attack_name', 'deepfool', 'Name of the attack (used for directory name)')
flags.DEFINE_integer('number_img', 1000, 'Number of images to attack')

flags.DEFINE_integer('max_iter', 10000, 'Maximum number of steps')
flags.DEFINE_float('step_size', 0.01, 'Relative step size')
flags.DEFINE_float('max_step_size', 0.25, 'Relative step size')
