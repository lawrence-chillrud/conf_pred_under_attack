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
import pandas as pd
import numpy as np
import os

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'
if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# %%
# Load CIFAR-100
train_ds, cal_ds, val_ds, test_ds = download_cifar100_as_ds()

# Prep CIFAR-100
train_ds, cal_ds, val_ds, test_ds = prep_data(train_ds, cal_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

# %%
# Initialize Model
model = init_model(freeze_base=False)
print(model.summary())

# Set optimizer
model.compile(
    optimizer=Adam(lr=ALPHA),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
# early stopping to monitor the validation loss and avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# reducing learning rate on plateau
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3, factor=0.1, min_lr=1e-6, verbose=1)

# %%
# Train the model on the CIFAR-100 dataset
hist = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=[early_stop, rlrop]
)

# %%
# Save model
model.save_weights(f'{OUTPUT_DIR}/weights.h5')
df = pd.DataFrame.from_dict(hist.history)
df['epoch'] = np.arange(1, df.shape[0] + 1)
df.to_csv(f'{OUTPUT_DIR}/training_history.csv')
print(f"Complete. Saved results to: {OUTPUT_DIR}")