"""
This module contains various methods, that are used
in different places in this project
"""

import os
import importlib
import inspect
import sys
import tensorflow as tf
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
            epoch = tf.get_variable('epoch', dtype=tf.int64, initializer=0, trainable=False)
            graph.add_to_collection('epoch_key', epoch)
            
            epoch_op = tf.assign_add(epoch, 1)
            graph.add_to_collection('epoch_key', epoch_op)
   
