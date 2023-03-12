# File: train.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Train model on CIFAR-100 for image classification

# %%
from download_cifar100 import download_cifar100_as_ds
from prep_data import prep_data
from models import init_model
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
# %%
# Load CIFAR-100
train_ds, cal_ds, val_ds, test_ds = download_cifar100_as_ds()

# Prep CIFAR-100
train_ds, cal_ds, val_ds, test_ds = prep_data(train_ds, cal_ds, val_ds, test_ds)

# %%
# Set GPU
gpus_available = tf.config.list_physical_devices('GPU')
if gpus_available: tf.config.set_visible_devices(gpus_available[5], 'GPU')

# %%
# Initialize Model
model = init_model()

# Set optimizer
model.compile(
    optimizer=Adam(lr=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# %%
# Train the model on the CIFAR-100 dataset
hist = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=20, 
    callbacks=[early_stop, rlrop]
)

# %%
