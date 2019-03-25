import tensorflow as tf

flags = tf.app.flags



flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_float('l2_reg', 0.0, 'L2 regularization scale. Set to 0 to disable')

flags.DEFINE_boolean('restore', False, 'If true, restore from last checkpoint')
flags.DEFINE_string('ckpt_dir', 'ckpt', 'checkpoint directory')
flags.DEFINE_string('model_pck', 'models', 'package name of models')

cfg = flags.FLAGS