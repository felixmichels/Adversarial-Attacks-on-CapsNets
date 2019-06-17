"""
Computes singular values of the matrix containing normalized
adversarial perturbations as norm.
Since DeepFool is the closest we have to fast, minimal adversarial perturbations,
so we're going to use DeepFool
"""

import sys
sys.path.insert(0,'..')

import numpy as np
import numpy.linalg as la
from util.data import get_adv, get_attack_original
from util.imgdataset import dataset_by_name

attack_names = ['deepfool', 'deepfool_more']

model_names = {
    'mnist': {'conv': 'ConvBaseline',
              'caps': 'CapsNetSmall',
              'linear': 'SimpleNet'
              },
    'fashion_mnist': {'conv': 'ConvBaseline',
                      'caps': 'CapsNetVariant',
                      'linear': 'SimpleNet'
                      },
    'svhn': {'conv': 'ConvBaseline',
             'caps': 'CapsNetVariant',
             'linear': 'SimpleNet'
             },
    'cifar10': {'conv': 'ConvGood',
                'caps': 'DCNet',
                'linear': 'SimpleNet'
                }
}


def filter_normalize_pert(A):
    """
    Filters out NaN values (failed attack),
    and perturbations near zero (already misclassified),
    then normalizes columns
    """
    epsilon = 1e-12

    A = A.reshape(A.shape[0], -1).T
    idx = np.isfinite(A).all(axis=0)
    A = A[:, idx]
    norms = la.norm(A, axis=0)

    idx = norms >= epsilon
    A[:, np.logical_not(idx)] = 0.0
    A[:, idx] /= norms[idx]

    return A


def save_sing(pert, dataset_name, model_type):
    A = filter_normalize_pert(pert)

    print('Compute svd for {} on {}\nPerturbation matrix shape: {}'.format(
        model_type, dataset_name, A.shape))

    sing = la.svd(A, compute_uv=False)
    np.save('singular_values_' + dataset_name + '_' + model_type, sing)


def main():
    np.random.seed(239048123)

    for dataset_name in model_names:
        dataset = dataset_by_name(dataset_name)

        orig = np.concatenate([get_attack_original(attack, dataset)[0] for attack in attack_names])

        rand_pert = np.random.normal(size=orig.shape)
        save_sing(rand_pert, dataset_name, 'random')

        for model_type in model_names[dataset_name]:
            model_name = model_names[dataset_name][model_type]

            adv = np.concatenate([get_adv(model_name, attack, dataset_name) for attack in attack_names])
            save_sing(orig-adv, dataset_name, model_type)


if __name__ == '__main__':
    main()
