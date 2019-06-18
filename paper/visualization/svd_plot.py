#!/usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt


def file_name(dataset_name, model_type):
    file_name = 'singular_values_' + dataset_name + '_' + model_type + '.npy'
    return os.path.join('../../code/data_collection/', file_name)


for dataset_name in 'mnist', 'fashion_mnist', 'svhn', 'cifar10':
    plt.figure()
    svd_caps = np.load(file_name(dataset_name, 'caps'))
    svd_conv = np.load(file_name(dataset_name, 'conv'))
    svd_random = np.load(file_name(dataset_name, 'random'))

    N = len(svd_random)

    plt.plot(svd_random, '--', label='random')
    plt.plot(svd_caps, label='CapsNet')
    plt.plot(svd_conv, label='ConvNet')

    plt.xlim((0, N))
    plt.ylim((0, 5))
    plt.xlabel('index')
    plt.ylabel('singular value')
    
    plt.legend()

    plt.savefig('../figures/svd_' + dataset_name + '.pdf')
