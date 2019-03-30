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



def load_cifar10(is_train=True, batch_size=None):
    training, test = _load_scaled_cifar10()
    x,y = training if is_train else test
    
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    if is_train:
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
        raise FileNotFoundError
        
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
    [(model_name, model_class)] = inspect.getmembers(model_module, 
                                                     lambda c: inspect.isclass(c) and sys.modules[c.__module__]==model_module)
      
    tf.logging.debug('Found class %s', model_class)
    return model_name, model_class