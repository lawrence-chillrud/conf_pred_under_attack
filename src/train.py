# File: train.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Train model on CIFAR-100 for image classification

# %%
from download_cifar100 import download_cifar100
# from models import init_model
# import tensorflow as tf

# %%
# Load CIFAR-100
(x_train, y_train), (x_cal, y_cal), (x_val, y_val), (x_test, y_test) = download_cifar100()
print(f'train: {x_train.shape} {y_train.shape}\ncal: {x_cal.shape} {y_cal.shape}, \nval: {x_val.shape} {y_val.shape}\ntest: {x_test.shape} {y_test.shape}')

# %%
# Set GPU
gpus_available = tf.config.list_physical_devices('GPU')
if gpus_available: tf.config.set_visible_devices(gpus_available[5], 'GPU')

# %%
# Initialize Model
model = PreTrainedEfficientNetV2B0()

# %%
# Train the model on the CIFAR-100 dataset
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# %%
