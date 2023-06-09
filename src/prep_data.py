# File: prep_data.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/12/2023
# Description: Prep CIFAR-100 datasets
import tensorflow as tf

def prep_data(train_ds, cal_ds, val_ds, test_ds, batch_size=8):

    def preprocess(x):
        # normalize
        x = tf.cast(x, tf.float16)
        x = tf.divide(x, 255.)

        # resize
        x = tf.image.resize(x, size=[224, 224], method='bilinear')
        x = tf.cast(x, tf.float16)
        return x
    
    def aug_mirror(x):
        return tf.image.random_flip_left_right(x)

    def aug_shift(x, w=4):
        y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    # Prep training data
    if train_ds is not None:
        train_ds = train_ds.map(lambda x, y: (preprocess(aug_shift(aug_mirror(x))), y)).batch(batch_size)
    if cal_ds is not None:
        cal_ds = cal_ds.map(lambda x, y: (preprocess(x), y)).batch(batch_size)
    if val_ds is not None:
        val_ds = val_ds.map(lambda x, y: (preprocess(x), y)).batch(batch_size)
    if test_ds is not None:
        test_ds = test_ds.map(lambda x, y: (preprocess(x), y)).batch(batch_size)
    
    return train_ds, cal_ds, val_ds, test_ds
