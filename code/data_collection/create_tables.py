#!/usr/bin/env python3

import numpy as np
import os
import sys
import numpy.linalg as la


def save_table(name, rows, columns, data, fmt='%2.2f'):
    file_ext = '.csv'
    row_count, column_count = data.shape
    assert row_count == len(rows) and column_count == len(columns)
    str_data = np.char.mod(fmt, data)

    try:
        rows = rows.split()
    except AttributeError:
        pass

    try:
        columns = columns.split()
    except AttributeError:
        pass

    rows = np.array(rows).reshape(-1, 1)
    header = '\t' + '\t'.join(columns)

    if not name.endswith(file_ext):
        name += file_ext

    np.savetxt(name, np.hstack((rows, str_data)),
               fmt='%s', header=header, delimiter='\t')


def average_norm(x, orig=None, filter='default'):
    """ Average norm

    Args:
        x: The perturbations if orig is None, otherwise
           the adversarial examples
        orig: The original images
        filter: Function that filters invalid perturbations,
                or string 'default' to use the function valid_index

    Returns:
        Mean and std of the norms of x or x-orig, if orig is not None
    """
    if orig is not None:
        N = min(len(orig), len(x))
        x = x[:N] - orig[:N]

    if filter is None:
        idx = np.ones(N, dtype=np.bool)
    elif filter == 'default':
        idx = valid_index(x)
    elif callable(filter):
        idx = filter(x)
    else:
        raise ValueError('Invalid filter')

    norms = np.array([la.norm(xi) for xi in x[idx] ])
    return norms.mean(), norms.std()


def valid_index(x):
    """ Boolean mask of images without NaN of inf """
    return np.isfinite(x).all(axis=tuple(range(1, x.ndim)))


model_names = {
    'mnist': {'conv': 'ConvBaseline',
              'caps': 'CapsNetSmall'
              },
    'fashion_mnist': {'conv': 'ConvBaseline',
                      'caps': 'CapsNetVariant'
                      },
    'svhn': {'conv': 'ConvBaseline',
             'caps': 'CapsNetVariant'
             },
    'cifar10': {'conv': 'ConvGood',
                'caps': 'DCNet'
                }
}

dataset_names = list(model_names.keys())

adv_numbers = {
    'carlini_wagner': 500,
    'boundary_attack': 1000,
    'deepfool': 1000,
    'universal_perturbation': 100
}
attack_names = list(adv_numbers.keys())

def targeted(name):
    return name == 'carlini_wagner'

def main(args):
    data_dir = '../data'
    if len(args) > 0:
        data_dir = args[0]

    for model in 'caps', 'conv':
        data = np.empty((len(attack_names), len(dataset_names)))
        for att_idx, attack_name in enumerate(attack_names):
            for dat_idx, dataset_name in enumerate(dataset_names):
                attack_path = os.path.join(data_dir, dataset_name, attack_name)
                with np.load(os.path.join(attack_path, 'originals.npz')) as npz:
                    orig = npz['img']

                model_dir = os.path.join(attack_path, model_names[dataset_name][model])
                adv_files = [f for f in os.listdir(model_dir) if f.endswith('.npy')]
                advs = np.concatenate([np.load(os.path.join(model_dir, f)) for f in adv_files])

                advs = advs[:adv_numbers[attack_name]]

                if attack_name == 'universal_perturbation':
                    value = average_norm(advs)
                else:
                    value = average_norm(advs, orig=orig)

                data[att_idx, dat_idx] = value[0]

        save_table('norms_'+model, attack_names, dataset_names, data)

    for model, other in (('caps', 'conv'), ('conv', 'caps')):
        data = np.empty((len(attack_names), len(dataset_names)))
        for att_idx, attack_name in enumerate(attack_names):
            for dat_idx, dataset_name in enumerate(dataset_names):
                attack_path = os.path.join(data_dir, dataset_name, attack_name)
                with np.load(os.path.join(attack_path, 'originals.npz')) as npz:
                    orig = npz['img']
                    label = npz['label']
                    if targeted(attack_name):
                        target = npz['target_label']

                model_name = model_names[dataset_name][model]
                source_name = model_names[dataset_name][other]
                path = os.path.join(data_dir,
                                    dataset_name,
                                    attack_name,
                                    'Measure_'+source_name+'_'+model_name)

                probs = np.concatenate([np.load(os.path.join(path, file)) for file in os.listdir(path)])

                preds = probs.argmax(axis=-1)
                preds = preds[:adv_numbers[attack_name]]

                if attack_name == 'universal_perturbation':
                    value = np.concatenate([(prob.argmax(axis=-1) != label) for prob in probs]).mean()
                elif targeted(attack_name):
                    value = (preds == target[:len(preds)]).mean()
                else:
                    value = (preds != label[:len(preds)]).mean()

                data[att_idx, dat_idx] = value

        data *= 100
        save_table('transfer_fooling_'+model, attack_names, dataset_names, data)


if __name__ == '__main__':
    main(sys.argv[1:])
