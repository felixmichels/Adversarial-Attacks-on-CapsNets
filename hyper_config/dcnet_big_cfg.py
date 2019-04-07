import tensorflow as tf
flags = tf.app.flags
cfg = flags.FLAGS

flags.DEFINE_float('l2_scale', 1e-6, 'Scale for l2 loss')
flags.DEFINE_float('recon_scale', 0.0015, 'Scale for reconstruction loss')
cfg.test_every_n = 2
cfg.data_aug=0.5
