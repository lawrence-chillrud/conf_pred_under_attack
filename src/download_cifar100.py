# File: download_cifar100.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Load the CIFAR-100 dataset and split it into training, calibration, validation, and testing data
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split

def download_cifar100_as_numpy(train_cal_size=0.8, cal_len=1000, seed=0, normalize=False):
    '''
    Usage
    -----
    `(x_train, y_train), (x_cal, y_cal), (x_val, y_val), (x_test, y_test) = download_cifar100_as_numpy()`
    `print(f'train: {x_train.shape} {y_train.shape}\ncal: {x_cal.shape} {y_cal.shape}, \nval: {x_val.shape} {y_val.shape}\ntest: {x_test.shape} {y_test.shape}')`
    '''
    # Load the CIFAR-100 dataset
    (x_train_cal_val, y_train_cal_val), (x_test, y_test) = cifar100.load_data()

    # Normalize the pixel values to the range [0, 1]
    if normalize:
        x_train_cal_val = x_train_cal_val / 255.0
        x_test = x_test / 255.0

    # Split train_cal_val into train, cal, and val datasets
    x_train_cal, x_val, y_train_cal, y_val = train_test_split(x_train_cal_val, y_train_cal_val, train_size=train_cal_size, stratify=y_train_cal_val, random_state=seed)
    train_size = 1 - cal_len/len(y_train_cal)
    x_train, x_cal, y_train, y_cal = train_test_split(x_train_cal, y_train_cal, train_size=train_size, stratify=y_train_cal, random_state=seed)

    return (x_train, y_train), (x_cal, y_cal), (x_val, y_val), (x_test, y_test)

def download_cifar100_as_ds(train_split=0.78, cal_split=0.02, val_split=0.2, shuffle=True, seed=0):
    assert train_split + cal_split + val_split == 1

    # _ is metadata
    ds, _ = tfds.load("cifar100", with_info=True, as_supervised=True)
    train_cal_val_ds = ds['train']
    test_ds = ds['test']

    train_cal_val_size = 50000 # hardcoding in the training data size
    if shuffle: train_cal_val_ds = train_cal_val_ds.shuffle(train_cal_val_size, seed=seed)
    
    train_size = int(train_split*train_cal_val_size)
    cal_size = int(cal_split*train_cal_val_size)
    
    train_ds = train_cal_val_ds.take(train_size)
    cal_ds = train_cal_val_ds.skip(train_size).take(cal_size)
    val_ds = train_cal_val_ds.skip(train_size + cal_size)
    
    return train_ds, cal_ds, val_ds, test_ds