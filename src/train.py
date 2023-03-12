# File: train.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Train model on CIFAR-100 for image classification

# %%
from download_cifar100 import download_cifar100_as_ds
from models import init_model
import tensorflow as tf

# %%
# Load CIFAR-100
train_ds, cal_ds, val_ds, test_ds = download_cifar100_as_ds()

# %%
# Set GPU
gpus_available = tf.config.list_physical_devices('GPU')
if gpus_available: tf.config.set_visible_devices(gpus_available[5], 'GPU')

# %%
# Initialize Model
model = init_model()

# %%
# Train the model on the CIFAR-100 dataset
hist = model.fit(train_ds, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# %%
