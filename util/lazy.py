#@misc{hafner2016scopedecorator,
#  author = {Hafner, Danijar},
#  title = {Structuring Your TensorFlow Models},
#  year = {2016},
#  howpublished = {Blog post},
#  url = {https://danijar.com/structuring-your-tensorflow-models/}
#}


import functools
import tensorflow as tf

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def lazy_scope_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            tf.logging.debug('Building graph node %s-%s', self.__class__.__name__, function.__name__)
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator