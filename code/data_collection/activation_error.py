#!/usr/bin/env python
# coding: utf-8

"""
This scripts measures activation errors for internal layers
and generates plots
"""

import os
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import cycle

out_dir = '../../paper/figures'

short_attack_names = {
        'carlini_wagner': 'CW',
        'deepfool': 'DeepFool',
        'boundary_attack': 'Boundary',
        'universal_perturbation': 'Universal'
}

attack_names = list(short_attack_names.keys())
        

important_layers = ['MaxPool', 'PrimaryCaps', 'ConvCaps']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
      
layer_name_maps = {  
    'conv': OrderedDict([
        ('0_Placeholder', 'input'),
        ('1_conv2d_Relu', 'conv1'),
        ('2_conv2d_1_Relu', 'conv2'),
        ('5_batch_normalization_FusedBatchNorm', 'maxpool1'),
        ('6_conv2d_2_Relu', 'conv3'),
        ('7_conv2d_3_Relu', 'conv4'),
        ('10_batch_normalization_1_FusedBatchNorm', 'maxpool2'),
        ('11_conv2d_4_Relu', 'conv5'),
        ('14_batch_normalization_2_FusedBatchNorm', 'maxpool3'),
        ('17_dropout_3_Identity', 'fc1'),
        ('18_dense_1_BiasAdd', 'class_scores')
    ]),

    'caps': OrderedDict([
        ('0_Placeholder', 'input'),
        ('18_batch_normalization_7_FusedBatchNorm', 'densely_connected'),
        ('20_encoder_strided_slice', 'primarycaps'),
        ('22_encoder_strided_slice_6', 'convcaps'),
        ('24_encoder_Squeeze', 'classcaps')
    ])
}


def load_activations(data_dir, attack_name, arch_type):
    if arch_type == 'caps':
        file_name = 'DCNet3_encoder.npz'
    elif arch_type == 'conv':
        file_name = 'ConvGood_logits.npz'
    path = os.path.join(data_dir, 'cifar10', attack_name, 'activations')
    adv_path = os.path.join(path, 'adversarial_' + file_name)
    orig_path = os.path.join(path, 'original_' + file_name)
    return np.load(orig_path), np.load(adv_path)
# In[10]:


def error(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return ((np.abs(x - y)).max(axis=-1) / np.abs(x).max(axis=-1)).mean()


def is_capsule_layer(x):
    # Shape is either NCD or NHWCD
    return x.ndim % 2 == 1


def activation_errors(orig, adv, name_map, caps=False):
    assert orig.files == adv.files, 'Original/Adversarial activations mismatch'

    result = OrderedDict()
    result_prob = OrderedDict()

    for v, k in name_map.items():
        x = orig[v]
        y = adv[v]

        assert x.shape == y.shape, v + ' shape mismatch'
        # Filter out NaN from batch dimension
        axis = tuple(range(1, x.ndim))
        idx = np.logical_and(
                np.isfinite(x).all(axis=axis),
                np.isfinite(y).all(axis=axis))
        x = x[idx]
        y = y[idx]

        result[k] = error(x, y)
        if caps:
            if x.ndim % 2 == 1:
                x_norm = np.sqrt((x**2).sum(axis=-1))
                y_norm = np.sqrt((y**2).sum(axis=-1))
                result_prob[k] = error(x_norm, y_norm)
            else:
                result_prob[k] = result[k]
    
    orig.close()
    adv.close()
    if caps:
        return result, result_prob
    return result


def plot_error(err, **kwargs):
    plt.plot(list(err.values()), **kwargs)
    ticks = []
    labels = []
    for x, name in enumerate(err):
        for l in important_layers:
            if l.lower() in name:
                labels.append(l)
                ticks.append(x)
                plt.axvline(x, c='lightgray', ls='--', lw=1.0)
                break
                
    plt.xticks(ticks, labels)
    plt.legend(loc='upper left')
    plt.ylabel('activation error')


def savefig(fname):
    fending = '.pdf'
    if not fname.endswith(fending):
        fname += fending
    plt.savefig(os.path.join(out_dir, fname))
    

def save_individual(errs, name):
    for attack in errs:
        plt.figure()
        err = errs[attack]
        if 'caps' in name:
            err1, err2 = err
            plot_error(err1, label='full activations')
            plot_error(err2, label='capsule norms')
        else:
            plot_error(err) 
        savefig('activation_error_'+name+'_'+attack)
        

def save_all(errs, name):
    plt.figure()
    colorc = cycle(colors)
    for attack in errs:
        color = next(colorc)
        err = errs[attack]
        if 'caps' in name:
            err1, err2 = err
            plot_error(err1, label=short_attack_names[attack], color=color)
            plot_error(err2, c=color, ls='--')
        else:
            plot_error(err, label=short_attack_names[attack], color=color)
    savefig('activation_error_'+name)
    
    
def main(args):
    np.random.seed(239048123)
    data_dir = '../data'
    if len(args) > 0:
        data_dir = args[0]
        
    caps_err = {}
    conv_err = {}
    
    for attack in attack_names:
        caps_err[attack] = activation_errors(
                *load_activations(data_dir, attack, 'caps'),
                layer_name_maps['caps'],
                caps=True)

        
        conv_err[attack] = activation_errors(
                *load_activations(data_dir, attack, 'conv'),
                layer_name_maps['conv'],
                caps=False)
                          
        
    save_individual(caps_err, 'caps')
    save_individual(conv_err, 'conv')
    save_all(caps_err, 'caps')
    save_all(conv_err, 'conv')
    
    del caps_err['universal_perturbation']
    del conv_err['universal_perturbation']
    
    save_all(caps_err, 'caps_no_universal')
    save_all(conv_err, 'conv_no_universal')
        
        
if __name__ == '__main__':
    main(sys.argv[1:])