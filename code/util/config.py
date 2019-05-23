import os
import sys
import tensorflow as tf
import __main__

flags = tf.app.flags

def _is_interactive():
    return not hasattr(__main__, '__file__')


flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_float('data_aug', None, 'Amount of data augmentation. Set to None to disable')
flags.DEFINE_float('aug_prob', 0.75, 'Probability, that data augmentation is applied')
flags.DEFINE_boolean('aug_flip', False, 'If training data should be randomly flipped')

flags.DEFINE_integer('classes', 11, 'Number of classes of dataset') # One garbage class

flags.DEFINE_boolean('restore', False, 'If true, restore from last checkpoint')

flags.DEFINE_string('ckpt_dir', 'ckpt', 'checkpoint directory')
flags.DEFINE_string('model_pck', 'models', 'package name of models')
flags.DEFINE_string('data_dir', '../data', 'directory for generated perturbations')
flags.DEFINE_string('log_dir', 'logdir', 'directory for graphs and summaries')
flags.DEFINE_string('dataset_dir', 'datasets', 'directory for training/test data')
flags.DEFINE_string('param_dir', 'parameters', 'directory for (hyper-)parameters')

flags.DEFINE_float('save_freq', 300.0, 'Saves after epoch, if time in seconds since last save surpasses this value')
flags.DEFINE_integer('save_every_n', 5, 'Save every n epochs')
flags.DEFINE_integer('train_log_every_n', 50, 'Log training information every n batches')
flags.DEFINE_integer('summary_every_n', 50, 'Write summary information every n batches')
flags.DEFINE_integer('test_every_n', 5, 'Test every n epochs')

flags.DEFINE_boolean('debug', False, 'Sets logging verbosity to debug')
flags.DEFINE_boolean('stop_before_session', False, 'For debugging purposes')
flags.DEFINE_boolean('no_summary', False, 'For testing')
flags.DEFINE_boolean('no_save', False, 'For testing')
flags.DEFINE_boolean('test', False, 'For testing')


flags.DEFINE_string('config_dir', 'configs', 'directory for config files')
flags.DEFINE_string('extra_cfg', '', 'extra config files')

if _is_interactive():
    flags.DEFINE_string('f', '', 'kernel')
else:
    code_dir = os.path.dirname(os.path.realpath(__main__.__file__))
    os.chdir(code_dir)
    tfcaps_dir = os.path.join(code_dir, 'tensorflow-capsules')
    sys.path.append(tfcaps_dir)


cfg = flags.FLAGS


if cfg.test:
    cfg.epochs = 0
    cfg.no_summary = True
    cfg.no_save = True
    cfg.restore = True

if cfg.debug:
    tf.logging.set_verbosity(tf.logging.DEBUG)
else:
    tf.logging.set_verbosity(tf.logging.INFO)


def _load_config(mod_name):
    mod_file = cfg.config_dir + '.' + mod_name + '_cfg'
    __import__(mod_file)
    tf.logging.info('Loaded additional config from %s', mod_file)

def load_config(mod_name, optional=False):
    if optional:
        try:
            _load_config(mod_name)
        except ModuleNotFoundError:
            tf.logging.debug('No config %s', mod_name)
            pass
    else:
        _load_config(mod_name)
    

mod_name = __main__.__file__.split('/')[-1][:-3] if not _is_interactive() else 'default'
load_config(mod_name, optional=True)

if cfg.extra_cfg != '':
    for m in cfg.extra_cfg.split(','):
        load_config(m)
