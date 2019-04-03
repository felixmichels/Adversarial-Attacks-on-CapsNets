import tensorflow as tf
import __main__

flags = tf.app.flags



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
flags.DEFINE_string('hyper_dir', 'hyper_config', 'directory for hyperparameter config files')

flags.DEFINE_float('save_freq', 300.0, 'Saves after epoch, if time in seconds since last save surpasses this value')
flags.DEFINE_integer('save_every_n', 5, 'Save every n epochs')
flags.DEFINE_integer('train_log_every_n', 50, 'Log training information every n batches')
flags.DEFINE_integer('summary_every_n', 50, 'Write summary information every n batches')
flags.DEFINE_integer('test_every_n', 5, 'Test every n epochs')

flags.DEFINE_boolean('debug', False, 'Sets logging verbosity to debug')
flags.DEFINE_boolean('stop_before_session', False, 'For debugging purposes')

flags.DEFINE_string('hyper_cfg', '', 'hyperparameter config file')

cfg = flags.FLAGS

try:
    script_name = __main__.__file__[:-3]+'_cfg'
    __import__(cfg.hyper_dir+'.'+script_name)
except ModuleNotFoundError:
    pass

if cfg.hyper_cfg != '':
    for m in cfg.hyper_cfg.split(','):
        __import__(cfg.hyper_dir+'.'+m)
