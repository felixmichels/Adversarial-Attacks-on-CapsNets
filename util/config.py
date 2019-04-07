import tensorflow as tf
import __main__

flags = tf.app.flags

def _is_interactive():
    return not hasattr(__main__, '__file__')


flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_float('data_aug', None, 'Amount of data augmentation. Set to None to disable')
flags.DEFINE_float('aug_prob', 0.75, 'Probability, that data augmentation is applied')

flags.DEFINE_integer('classes', 11, 'Number of classes of dataset') # One garbage class

flags.DEFINE_boolean('restore', False, 'If true, restore from last checkpoint')

flags.DEFINE_string('ckpt_dir', 'ckpt', 'checkpoint directory')
flags.DEFINE_string('model_pck', 'models', 'package name of models')
flags.DEFINE_string('data_dir', 'data', 'directory for generated perturbations')
flags.DEFINE_string('log_dir', 'logdir', 'directory for graphs and summaries')

flags.DEFINE_float('save_freq', 300.0, 'Saves after epoch, if time in seconds since last save surpasses this value')
flags.DEFINE_integer('save_every_n', 5, 'Save every n epochs')
flags.DEFINE_integer('train_log_every_n', 50, 'Log training information every n batches')
flags.DEFINE_integer('summary_every_n', 50, 'Write summary information every n batches')
flags.DEFINE_integer('test_every_n', 5, 'Test every n epochs')

flags.DEFINE_boolean('debug', False, 'Sets logging verbosity to debug')
flags.DEFINE_boolean('stop_before_session', False, 'For debugging purposes')

flags.DEFINE_string('hyper_dir', 'hyper_config', 'directory for hyperparameter config files')
flags.DEFINE_string('hyper_cfg', '', 'hyperparameter config file')

if _is_interactive():
    flags.DEFINE_string('f', '', 'kernel')

cfg = flags.FLAGS

if cfg.debug:
    tf.logging.set_verbosity(tf.logging.DEBUG)
else:
    tf.logging.set_verbosity(tf.logging.INFO)

def load_config(mod_name):
    __import__(cfg.hyper_dir + '.' + mod_name + '_cfg')
    tf.logging.debug('Loaded additional config from %s', mod_name)

try:
    mod_name = __main__.__file__[:-3] if not _is_interactive() else 'default'
    load_config(mod_name)
except ModuleNotFoundError:
    pass

if cfg.hyper_cfg != '':
    for m in cfg.hyper_cfg.split(','):
        load_config(m)
