# File: test.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/13/2023
# Description: Test model on CIFAR-100 for image classification
# %%
from download_cifar100 import download_cifar100_as_ds
from prep_data import prep_data
from models import init_model
import tensorflow as tf
from keras.optimizers import Adam

# Hyperparameters
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'

# Load CIFAR-100
train_ds, cal_ds, val_ds, test_ds = download_cifar100_as_ds()

# Prep CIFAR-100
train_ds, cal_ds, val_ds, test_ds = prep_data(train_ds, cal_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

# Load in model
model = init_model(freeze_base=False)
model.load_weights(f'{OUTPUT_DIR}/weights.h5')
print(model.summary())

# Test performance
model.compile(
    optimizer=Adam(lr=ALPHA),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
loss, acc = model.evaluate(test_ds)
print(f'Loss: {loss}, Accuracy: {acc}')
# %%
