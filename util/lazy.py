# @misc{hafner2016scopedecorator,
#  author = {Hafner, Danijar},
#  title = {Structuring Your TensorFlow Models},
#  year = {2016},
#  howpublished = {Blog post},
#  url = {https://danijar.com/structuring-your-tensorflow-models/}
# }

"""
This module contains decorators for lazy properties,
designed to work with tensorflow
"""

import functools
import tensorflow as tf


def lazy_property(function):
    """
    Decorator, that initializes a property the first time it's called,
    and uses a cached value every time after that
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def lazy_scope_property(function, only_training=False):
    """
    Only to be used within models.basicmodel and subclasses
    Don't read, it's much too ugly
    """

    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if self.is_build:
            if only_training:
                self.training_vars.append(function.__name__)
            else:
                self.normal_vars.append(function.__name__)
            return None

        if not self.trainable and only_training:
            raise NotImplementedError("Property {} only exists if the model is trainable".format(function.__name__))

        if not hasattr(self, attribute):
            tf.logging.debug('Building graph node %s-%s', self.__class__.__name__, function.__name__)
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
