"""
Saves some adversarial images as examples
"""

import sys
sys.path.insert(0,'..')


import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from collections import namedtuple
from util.data import get_adv, get_attack_original
from util.imgdataset import dataset_by_name


NUM_IMGS = 20
IMG_DIR = 'imgs'

attack_names = ['deepfool', 'universal_perturbation', 'boundary_attack', 'carlini_wagner']

num_adv = {
        'deepfool': 1000,
        'boundary_attack': 1000,
        'carlini_wagner': 500,
        'universal_perturbation': 10000
        }


model_names = {
    'mnist': {'conv': 'ConvBaseline',
              'caps': 'CapsNetSmall',
              },
    'fashion_mnist': {'conv': 'ConvBaseline',
                      'caps': 'CapsNetVariant',
                      },
    'svhn': {'conv': 'ConvBaseline',
             'caps': 'CapsNetVariant',
             },
    'cifar10': {'conv': 'ConvGood',
                'caps': 'DCNet',
                }
}


def scale_pert(pert):
    return np.clip(pert + 0.5, 0, 1)


def scale_pert_visible(pert):
    # Pert is in [-1,1]
    pert = pert.copy()

    max_val = np.abs(pert).max()
    pert /= max_val
    # Pert is still in [-1, 1]
    # but either -1 or 1 is reached
    pert = (pert + 1) / 2
    return pert


def filter_idx(pert):
    """
    Returns index, that filters out NaN values (failed attack),
    and perturbations near zero (already misclassified),
    """
    epsilon = 1e-14

    idx = np.isfinite(pert).all(axis=(1, 2, 3))
    
    norms = np.array([la.norm(p) for p in pert])

    idx = np.logical_and(norms >= epsilon, idx)
    return idx


def imsave(name, type, num, arr):
    fname = '_'.join([name, type, str(num)]) + '.png'
    fname = os.path.join(IMG_DIR, fname)
    plt.imsave(fname, arr, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)

def main():
    np.random.seed(239048123)
    
    imgs = {}
    AdvImg = namedtuple('AdvImg', ('orig', 'pert', 'adv'))
    
    for dataset_name in model_names:
        dataset = dataset_by_name(dataset_name)
        for attack_name in attack_names:

            orig = get_attack_original(attack_name, dataset)[0]
            orig = orig[:num_adv[attack_name]]

            for model_type in model_names[dataset_name]:
                model_name = model_names[dataset_name][model_type]
     
                adv = get_adv(model_name, attack_name, dataset_name)
                if 'universal' in attack_name:
                    pert = adv
                    orig = orig[:len(pert)]
                    adv = np.clip(orig + pert, 0, 1)
                else:
                    adv = adv[:num_adv[attack_name]]
                    pert = orig - adv

                imgs['_'.join([dataset_name, attack_name, model_type])] = \
                    AdvImg(orig, pert, adv)
     
    # Filter out stuff
    for dataset_name in model_names:
        for attack_name in attack_names:
            names = ['_'.join([dataset_name, attack_name, model_type]) for model_type in model_names[dataset_name]]
            idx = np.logical_and(*[filter_idx(imgs[name].pert) for name in names])

            for name in names:
                v = imgs[name]
                imgs[name] = AdvImg(
                    orig=v.orig[idx].squeeze(),
                    pert=v.pert[idx].squeeze(),
                    adv=v.adv[idx].squeeze())
        
    for name, img in imgs.items():
        for n in range(NUM_IMGS):
            imsave(name, 'orig', n, img.orig[n])
            imsave(name, 'adv', n, img.adv[n])
            
            imsave(name, 'pert', n, scale_pert(img.pert[n]))
            imsave(name, 'pertvisible', n, scale_pert_visible(img.pert[n]))


if __name__ == '__main__':
    main()
