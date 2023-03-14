# File: download_cifar100.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Load the CIFAR-100 dataset and split it into training, calibration, validation, and testing data
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

def download_cifar100_as_ds(train_split=0.78, cal_split=0.02, val_split=0.2, shuffle=True, seed=0):
    assert train_split + cal_split + val_split == 1

    ds, info = tfds.load("cifar100", with_info=True, as_supervised=True)
    train_cal_val_ds = ds['train']
    test_ds = ds['test']

    train_cal_val_size = 50000 # hardcoding in the training data size
    if shuffle: train_cal_val_ds = train_cal_val_ds.shuffle(train_cal_val_size, seed=seed)
    
    train_size = int(train_split*train_cal_val_size)
    cal_size = int(cal_split*train_cal_val_size)
    
    train_ds = train_cal_val_ds.take(train_size)
    cal_ds = train_cal_val_ds.skip(train_size).take(cal_size)
    val_ds = train_cal_val_ds.skip(train_size + cal_size)
    
    return train_ds, cal_ds, val_ds, test_ds, info.features['label'].names