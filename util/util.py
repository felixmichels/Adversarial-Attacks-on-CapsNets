import tensorflow as tf

def load_cifar10(is_train=True, batch_size=None):
    training, test = tf.keras.datasets.cifar10.load_data()
    x,y = training if is_train else test
    x = x / 255.0
    y = y.squeeze()
    y = y.astype('uint8')
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    if is_train:
        dataset = dataset.shuffle(2**13)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset