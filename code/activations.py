"""
Script for saving the activations of hidden layers
for original and adversarial images
"""

import os
import sys
import tensorflow as tf
import numpy as np
from util.util import get_model, get_dir, get_params
from util.data import get_adv, get_attack_original
from util.config import cfg
from util.imgdataset import dataset_by_name

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tensorflow-capsules'))
import tfcaps as tc


_io_dict = {}

def register_io(a):
    scope = tf.get_default_graph().get_name_scope()
    scope = scope.split('/')
    # don't need unnecessary scopes,
    # model name and conscruct (encoder/decoder is enough)
    scope = '_'.join([scope[0], scope[-1]])
    _io_dict[scope] = a


def _nice_tensor_name(v):
    name = getattr(v, 'name', 'unknown_object')
    name = name.split('/')[-2:]
    name = '_'.join(name)
    # Number at the beginning is enough,
    # delete the rest
    name = name[:name.rfind(':')]

    return name


def get_ios():
    return {
            scope: {
                str(n)+'_'+_nice_tensor_name(v): v
                for n, v in enumerate(a)
                }
            for scope, a in _io_dict.items()
            }


_normal_new_io = tc.layers.new_io


def new_io_with_act(inputs, return_list=False):
    """
    Wraps around tfcaps.new_io, to get to the activations
    """
    a, i, o = _normal_new_io(inputs, return_list=True)

    register_io(a)

    if return_list:
        return a, i, o
    return i, o


# Hijack the new_io method
tc.layers.new_io = new_io_with_act


def save_activations(sess, feed_dict, directory='.', file_prefix=''):
    activations = sess.run(get_ios(), feed_dict=feed_dict)

    for scope, acts in activations.items():
        file = os.path.join(directory, file_prefix + scope)
        np.savez(file, **acts)


def attack_activations(sess, attack_name, dataset, model):
    tf.logging.info('Activations for %s', attack_name)

    orig, _ = get_attack_original(attack_name, dataset)
    adv = get_adv(model.name, attack_name, dataset.name)

    orig = orig[:cfg.batch_size]

    if 'universal' in attack_name.lower():
        adv = orig + adv[0]
    else:
        adv = adv[:cfg.batch_size]

    directory = get_dir(cfg.data_dir, dataset.name, attack_name, 'activations')

    tf.logging.info('Measuring originals')
    save_activations(sess, feed_dict={model.img: orig, model.label: [0]}, directory=directory, file_prefix='original_')
    tf.logging.info('Measuring adversarial examples')
    save_activations(sess, feed_dict={model.img: adv, model.label: [0]}, directory=directory, file_prefix='adversarial_')


def main(args):

    model_name = args[1]
    model_class = get_model(model_name)
    dataset = dataset_by_name(args[2])

    if len(args) < 4:
        attacks = ['carlini_wagner', 'boundary_attack', 'deepfool', 'universal_perturbation']
    else:
        attacks = args[4].split(',')

    params = get_params(model_name, dataset.name)

    tf.logging.debug('Creating model graph')
    model = model_class(trainable=False, num_classes=dataset.num_classes, shape=dataset.shape, **params)

    tf.logging.debug('ios: %s', get_ios())

    ckpt_dir = get_dir(cfg.ckpt_dir, dataset.name, model.name)
    saver = tf.train.Saver(var_list=tf.global_variables(scope=model.scope))

    if cfg.stop_before_session:
        exit()

    tf.logging.debug('Starting session')
    with tf.Session() as sess:
        try:
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver.restore(sess, save_path)
            for attack in attacks:
                attack_activations(sess, attack, dataset, model)

        except KeyboardInterrupt:
            print("Manual interrupt occurred")


if __name__ == '__main__':
    tf.app.run()
