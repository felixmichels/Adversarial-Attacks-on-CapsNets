"""
This module contains various methods, that are used
in different places in this project
"""

import os
import importlib
import inspect
import sys
import json
import tensorflow as tf
import numpy as np
from util.config import cfg


def get_dir(base, *args):
    """
    Checks if base exists and is a directory.
    If it is, return base/arg0/arg1..., creating the dirs if necessary
    """
    if not os.path.isdir(base):
        tf.logging.fatal("%s path does not exist", base)
        raise FileNotFoundError("{} does not exist".format(base))

    path = os.path.join(base, *args)
    os.makedirs(path, exist_ok=True)

    return path


def get_model(name):
    """
    Loads the model class (in cfg.model_pck) based on the file name
    (case insensitive)
    """
    # Evil reflection
    model_name = name.lower()
    model_module = importlib.import_module('.'+model_name, cfg.model_pck)
    [(_, model_class)] = inspect.getmembers(
        model_module,
        lambda c: inspect.isclass(c) and sys.modules[c.__module__] == model_module)

    tf.logging.debug('Found class %s', model_class)
    return model_class


def _update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_params(name, *optionals):
    """
    Loads parameters from a json file,
    that may not exists.

    Args:
        name: name, that starts the filename
        *optionals: optional names, that are chained together to form the filename

    Returns:
        Returns a dict from
            name_optional[0]_optional[1]....json if it exists,
            name.json, if above does not exist
        If neither exists, returns an empty dict
        (combines possible dicts)
    """
    name = name.lower()
    optionals = [opt.lower() for opt in optionals]

    partial = os.path.join(cfg.param_dir, name+'.json')
    full = os.path.join(cfg.param_dir, '_'.join([name]+optionals)+'.json')

    param = {}
    if os.path.isfile(partial):
        with open(partial, 'r') as f:
            tf.logging.info('Loading parameters from %s', partial)
            _update(param, json.load(f))
    if os.path.isfile(full):
        with open(full, 'r') as f:
            tf.logging.info('Loading parameters from %s', full)
            _update(param, json.load(f))

    if not param:
        tf.logging.info('No parameter file found')
    return param


def func_with_prob(f, p):
    """
    Applies a function randomly

    Parameters:
        f: The function to apply
        p: Probability, that the function should be applied

    Returns:
        A function, that returns for input x
        f(x) with probability p,
        or x with probability 1-p
    """
    def g(x):
        choice = tf.random_uniform(shape=(), minval=0, maxval=1)
        return tf.cond(choice < p, lambda: f(x), lambda: x)
    return g


def get_epoch(graph=None):
    """Get global epoch counter"""
    graph = graph or tf.get_default_graph()
    
    epoch_tensors = graph.get_collection('epoch_key')
    if len(epoch_tensors) == 2:
        epoch = epoch_tensors[0]
    else:
        epoch = None
    return epoch


def get_epoch_op(graph=None):
    """Get global epoch update op"""
    graph = graph or tf.get_default_graph()
    
    epoch_tensors = graph.get_collection('epoch_key')
    if len(epoch_tensors) == 2:
        epoch_op = epoch_tensors[1]
    else:
        epoch_op = None
    return epoch_op


def create_epoch(graph=None):
    """Creates the global epoch counter"""
    if get_epoch(graph) is not None:
        raise ValueError('"Epoch" already exists')
        
    graph = graph or tf.get_default_graph()
    
    with tf.device('cpu:0'):
        with graph.as_default():
            epoch = tf.get_variable('epoch', dtype=tf.int64, initializer=tf.constant(0, dtype=tf.int64), trainable=False)
            graph.add_to_collection('epoch_key', epoch)
            
            epoch_op = tf.assign_add(epoch, 1)
            graph.add_to_collection('epoch_key', epoch_op)
   
    
def _bak_name(file, n):
    name = file
    if n > 0:
        name += '.bak'
    if n > 1:
        name += str(n)
    return name     


def np_save_bak(file, arr, num_bak=5):
    """numpy saves an array, but creates backups"""
    for i in range(num_bak, 0, -1):
        old = _bak_name(file, i-1)
        if os.path.isfile(old):
            os.rename(old, _bak_name(file, i))
    np.save(file, arr)
