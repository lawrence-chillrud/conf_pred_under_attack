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
import numpy as np
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'

# Load CIFAR-100
_, cal_ds, _, test_ds = download_cifar100_as_ds()

# Prep CIFAR-100
_, cal_ds, _, test_ds = prep_data(None, cal_ds, None, test_ds, batch_size=128)

# Load in model
model = init_model(freeze_base=False)
model.load_weights(f'{OUTPUT_DIR}/weights.h5')
print(model.summary())

model.compile(
    optimizer=Adam(lr=ALPHA),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# Test performance
loss, acc = model.evaluate(test_ds)
print(f'Loss: {loss}, Accuracy: {acc}')

# %%
# Extract softmax scores and labels from test and cal datasets
test_softmax_scores = np.empty((0, 100))
test_labels = np.empty(0)
n = tf.data.experimental.cardinality(test_ds)
for i, (x, y) in tqdm(enumerate(test_ds), total=n.numpy()):
    y_hat = model(x)
    test_softmax_scores = np.vstack((test_softmax_scores, y_hat.numpy()))
    test_labels = np.concatenate((test_labels, y.numpy()), axis=0)

test_labels = test_labels.astype(np.uint8)
print("Test:")
print(test_softmax_scores.shape)
print(test_softmax_scores)

cal_softmax_scores = np.empty((0, 100))
cal_labels = np.empty(0)
n = tf.data.experimental.cardinality(cal_ds)
for i, (x, y) in tqdm(enumerate(cal_ds), total=n.numpy()):
    y_hat = model(x)
    cal_softmax_scores = np.vstack((cal_softmax_scores, y_hat.numpy()))
    cal_labels = np.concatenate((cal_labels, y.numpy()), axis=0)

cal_labels = cal_labels.astype(np.uint8)
print("Cal:")
print(cal_softmax_scores.shape)
print(cal_softmax_scores)

# %%
