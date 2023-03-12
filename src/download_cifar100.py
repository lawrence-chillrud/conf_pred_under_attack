# File: download_cifar100.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Load the CIFAR-100 dataset and split it into training, calibration, validation, and testing data
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split

def download_cifar100(train_cal_size=0.8, cal_len=1000, seed=0):
    # Load the CIFAR-100 dataset
    (x_train_cal_val, y_train_cal_val), (x_test, y_test) = cifar100.load_data()

    # Normalize the pixel values to the range [0, 1]
    x_train_cal_val = x_train_cal_val / 255.0
    x_test = x_test / 255.0

    # Split train_cal_val into train, cal, and val datasets
    x_train_cal, x_val, y_train_cal, y_val = train_test_split(x_train_cal_val, y_train_cal_val, train_size=train_cal_size, stratify=y_train_cal_val, random_state=seed)
    train_size = 1 - cal_len/len(y_train_cal)
    x_train, x_cal, y_train, y_cal = train_test_split(x_train_cal, y_train_cal, train_size=train_size, stratify=y_train_cal, random_state=seed)

    return (x_train, y_train), (x_cal, y_cal), (x_val, y_val), (x_test, y_test)

# one hot encode labels
# y_train = tf.one_hot(y_train, depth=y_train.max() + 1, dtype=tf.float64)
# y_val = tf.one_hot(y_val, depth=y_val.max() + 1, dtype=tf.float64)

# y_train = tf.squeeze(y_train)
# y_val = tf.squeeze(y_val)