import tensorflow as tf
import numpy as np
from util.config import cfg
import os
import importlib
import inspect
import sys

def _load_scaled_cifar10():
    if _load_scaled_cifar10.data is None:
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        
        train_x = train_x.astype('float32')
        train_x /= 255.0
        train_y = train_y.squeeze().astype('int64')
        
        test_x = test_x.astype('float32')
        test_x /= 255.0
        test_y = test_y.squeeze().astype('int64')
        
        _load_scaled_cifar10.data = (train_x, train_y), (test_x, test_y)

    return _load_scaled_cifar10.data

_load_scaled_cifar10.data = None


def rand_crop_resize(min_size, prob_crop):
    def crop(img):
        crop_size = tf.random_uniform(shape=(), minval=min_size, maxval=1)
        x = tf.random_uniform(shape=(), minval=0, maxval=1-crop_size)
        y = tf.random_uniform(shape=(), minval=0, maxval=1-crop_size)
        box = [[x,y,x+crop_size,y+crop_size]]
        return tf.image.crop_and_resize([img], boxes=box, box_ind=[0], crop_size=[32,32])[0]
    
    def aug(img):
        choice = tf.random_uniform(shape=(), minval=0, maxval=1)
        return tf.cond(choice<prob_crop, lambda: crop(img), lambda: img)
    
    return aug
    

def chain_augs(*augs):
    def aug(x,y):
        for f in augs:
            x = f(x)
        x = tf.clip_by_value(x,0,1)
        return x,y
    return aug

def load_cifar10(is_train=True, batch_size=None):
    training, test = _load_scaled_cifar10()
    x,y = training if is_train else test
    
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    if is_train:
        dataset = dataset.map(
                chain_augs(
                        lambda x: tf.contrib.image.rotate(x, np.pi/15 * tf.random_normal(shape=())),
                        tf.image.random_flip_left_right,
                        lambda x: tf.image.random_hue(x, 0.05),
                        lambda x: tf.image.random_saturation(x, 0.6, 1.6),
                        lambda x: tf.image.random_brightness(x, 0.05),
                        lambda x: tf.image.random_contrast(x, 0.7, 1.3),
                        rand_crop_resize(0.6,0.6)
                        ),
                num_parallel_calls=32)
        dataset = dataset.shuffle(4*batch_size if batch_size is not None else 1024)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset

def get_dir(base, *args):
    """
    Checks if base exists and is a directory.
    If it is, return base/arg0/arg1..., creating the dirs if necessary
    """
    if not os.path.isdir(base):
        tf.logging.fatal("%s path does not exist", base)
        raise FileNotFoundError("{} does not exist".format(base))
        
    path = os.path.join(base,*args)
    os.makedirs(path, exist_ok=True)
    
    return path

def get_data(attack_name, n=None, override=False):
    """
    Loads original images from file, or creates file with random test images.
    n: number of images. If image file already exists, None will load all.
       If file doesn't exists yet, or override is True, n must not be None
    override: If true, generate a new file
    """
    path = get_dir(cfg.data_dir, attack_name)
    file_name = os.path.join(path, 'originals.npz')
    
    if not os.path.isfile(file_name) or override:
        _, (img, label) = _load_scaled_cifar10()
      
        idx = np.random.permutation(len(img))
        img = img[idx]
        label = label[idx]
        
        img = img[:n]
        label = label[:n]
        
        np.savez(file_name, img=img, label=label)
        
    else:
        tf.logging.debug('Loading from file %s', file_name)
        with np.load(file_name) as npzfile:
            img = npzfile['img']
            label = npzfile['img']
            
            img = img[:n]
            label = img[:n]
    
    return img,label


def get_model(name):
      # Evil reflection
    model_name = name.lower()
    model_module = importlib.import_module('.'+model_name,cfg.model_pck)
    [(_, model_class)] = inspect.getmembers(model_module, 
                                                     lambda c: inspect.isclass(c) and sys.modules[c.__module__]==model_module)
      
    tf.logging.debug('Found class %s', model_class)
    return model_class