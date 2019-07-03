#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:55:46 2019

@author: felix
"""

import os
from subprocess import check_call

IMG_WIDTH = 224
IN_DIR = 'imgs'
OUT_DIR = '../../paper/figures'

attack_names = ['deepfool', 'universal_perturbation', 'boundary_attack', 'carlini_wagner']
dataset_names = ['mnist', 'fashion_mnist', 'svhn', 'cifar10']
model_types = ['conv', 'caps']


def save_grid(outfile, width, height, *infiles):
    outfile = os.path.join(OUT_DIR, outfile)
    outfile_tmp = outfile + '.tmp'
    scale = '{}x{}'.format(IMG_WIDTH, IMG_WIDTH)
    tile = '{}x{}'.format(width, height)
    opts = ['-tile', tile, '-geometry', '+4+4', '-scale', scale]
    check_call(["montage"] + list(infiles) + opts + ['pdf:'+outfile_tmp])

    # Change pdf metadata, so git doesn't freak out
    stampfile = os.path.join(OUT_DIR, 'pdftimestamp.txt')
    check_call(['pdftk', outfile_tmp, 'update_info', stampfile, 'output', outfile])
    os.remove(outfile_tmp)


def img_file_name(dataset, attack, model_type, img_type, num):
    name = '_'.join([dataset, attack, model_type])
    fname = '_'.join([name, img_type, str(num)]) + '.png'
    return os.path.join(IN_DIR, fname)


def main():
    for attack in attack_names:
        for model_type in model_types:
            # Small cifar10 images
            infiles = [img_file_name('cifar10', attack, model_type, img_type, 1)
                     for img_type in ('adv', 'pertvisible')]
            outfile = '_'.join(['cifar10', attack, model_type + '.pdf'])
            save_grid(outfile, 2, 1, *infiles)
            
            infile = img_file_name('cifar10', attack, model_type, 'orig', 1)
            outfile = '_'.join(['cifar10', attack, 'orig.pdf'])
            save_grid(outfile, 1, 1, infile)
            

            # Big appendix image
            infiles = [img_file_name(dataset, attack, model_type, img_type, num)
                       for dataset in dataset_names
                       for num in (2, 3)
                       for img_type in ('adv', 'pertvisible')]

            outfile = '_'.join([attack, model_type, 'appendix.pdf'])
            save_grid(outfile, 2, 2*len(dataset_names), *infiles)
            
            infiles = [img_file_name(dataset, attack, model_type, 'orig', num)
                       for dataset in dataset_names
                       for num in (2, 3)]
            outfile = '_'.join([attack, 'orig', 'appendix.pdf'])
            save_grid(outfile, 1, 2*len(dataset_names), *infiles)


if __name__ == '__main__':
    main()
