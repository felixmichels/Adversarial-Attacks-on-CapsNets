import tensorflow as tf
flags = tf.app.flags
cfg = flags.FLAGS

flags.DEFINE_float('l2_scale', 1e-4, 'Scale for l2 loss')
cfg.data_aug=0.5
