# File: test.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/13/2023
# Description: Test model on CIFAR-100 for image classification
# %%
from download_cifar100 import download_cifar100_as_ds
from prep_data import prep_data
from models import init_model
from keras.optimizers import Adam
from extract_softmax_scores import extract_softmax_scores
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from reliability_diagrams import *

# Hyperparameters
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'

# Load and prep CIFAR-100
_, cal_ds, _, test_ds = download_cifar100_as_ds()
_, cal_ds, _, test_ds = prep_data(None, cal_ds, None, test_ds, batch_size=128)

# Load in model
model = init_model(freeze_base=False)
model.load_weights(f'{OUTPUT_DIR}/weights.h5')
print(model.summary())
model.compile(optimizer=Adam(lr=ALPHA), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
# Calibration set performance
loss, acc = model.evaluate(cal_ds)
print(f'Calibration set performance: loss {loss}, accuracy {acc}')

# %%
# Test performance
loss, acc = model.evaluate(test_ds)
print(f'Test set performance: loss {loss}, accuracy {acc}')

# %%
# Extract softmax scores and labels from cal and test datasets
cal_smx, cal_labels_dict = extract_softmax_scores(model, cal_ds)
cal_df = pd.DataFrame.from_dict(cal_labels_dict)

test_smx, test_labels_dict = extract_softmax_scores(model, test_ds)
test_df = pd.DataFrame.from_dict(test_labels_dict)

# %%
calibration_plot_data = OrderedDict()
calibration_plot_data['Calibration set'] = cal_labels_dict
calibration_plot_data['Test set'] = test_labels_dict

plt.style.use("seaborn")
plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

# %%
reliability_diagram(
    test_labels_dict['true_labels'], 
    test_labels_dict['pred_labels'], 
    test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='Unperturbed test set performance (no conformal prediction)', figsize=(6, 6), dpi=100, return_fig=True
)
# %%
