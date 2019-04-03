#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:57:45 2019

@author: felix
"""

import tensorflow as tf
flags = tf.app.flags

flags.DEFINE_string('attack_name', 'boundary_attack', 'Name of the attack (used for directory name)')
flags.DEFINE_integer('number_img', 1000, 'Number of images to attack')
flags.DEFINE_integer('processes', 32, 'Number of attacks to run parallel')

flags.DEFINE_float('eps_min', 1e-3, 'Epsilon min for the attack algorithm')
flags.DEFINE_integer('max_steps', 1000, 'Maximum number of evaluations in the attack algorithm')
