# File: test.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/13/2023
# Description: Test model on CIFAR-100 for image classification
# %%
from download_cifar100 import download_cifar100_as_ds
from prep_data import prep_data
from models import *
from keras.optimizers import Adam
from extract_softmax_scores import extract_softmax_scores
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from reliability_diagrams import *
from conformal_prediction import RAPS_conformal_prediction
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import tensorflow as tf

# Hyperparameters
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'

# Load and prep CIFAR-100
train_ds, cal_ds, val_ds, test_ds, human_readable_labels = download_cifar100_as_ds()
train_ds, cal_ds, val_ds, test_ds = prep_data(train_ds, cal_ds, val_ds, test_ds, batch_size=128)

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
# Extract softmax scores and labels from val and train datasets
val_smx, val_labels_dict = extract_softmax_scores(model, val_ds)
val_df = pd.DataFrame.from_dict(val_labels_dict)

train_smx, train_labels_dict = extract_softmax_scores(model, train_ds)
train_df = pd.DataFrame.from_dict(train_labels_dict)

# %%
plt.style.use("seaborn")
plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

reliability_diagram(
    test_labels_dict['true_labels'], 
    test_labels_dict['pred_labels'], 
    test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='Unperturbed test set performance (no conformal prediction)', figsize=(6, 6), dpi=100, return_fig=True
)

# %%
empirical_coverage, qhat, prediction_sets = RAPS_conformal_prediction(
    cal_smx=cal_smx, cal_labels=cal_labels_dict['true_labels'],
    val_smx=test_smx, val_labels=test_labels_dict['true_labels'],
    lam_reg=0.01, k_reg=5, rand=False
)

def pred_set_sizes_hist(prediction_sets, cov):
    sizes = np.sum(prediction_sets, axis=1)
    bins = np.arange(np.min(sizes), np.max(sizes)+1)
    plt.hist(sizes, bins=bins, edgecolor='black')
    plt.xlabel('RAPS CP prediction set size')
    plt.ylabel('Frequency')
    plt.title(f'Unperturbed test set empirical coverage: {cov}')
    plt.show()

pred_set_sizes_hist(prediction_sets, empirical_coverage)

# TODO PLOT IMAGES WITH THEIR PREDICTION SETS

# %%
# Construct adversarial attacks

# Load in model again for logit output
logits_model = load_logit_model(f'{OUTPUT_DIR}/weights.h5')
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
logits_model.compile(optimizer=Adam(lr=ALPHA), loss=loss_fn, metrics=['accuracy'])

# %%
# FGSM attack
fgsm_test_ds = test_ds.unbatch().take(BATCH_SIZE*10).map(
    lambda x, y: (tf.squeeze(fast_gradient_method(
        logits_model, tf.expand_dims(x, axis=0), 
        eps=0.01, norm=np.inf, targeted=False
    )), y)
).batch(BATCH_SIZE)

# %%
# PGD attack
pgd_test_ds = test_ds.map(
    lambda x, y: (tf.squeeze(projected_gradient_descent(
        logits_model, tf.expand_dims(x, axis=0), 
        eps=0.01, eps_iter=0.01, nb_iter=20,
        norm=np.inf, loss_fn=loss_fn, targeted=False
    )), y)
)

# %%
fgsm_test_smx, fgsm_test_labels_dict = extract_softmax_scores(model, fgsm_test_ds)
fgsm_test_df = pd.DataFrame.from_dict(fgsm_test_labels_dict)

# %%
pgd_test_smx, pgd_test_labels_dict = extract_softmax_scores(model, pgd_test_ds)
pgd_test_df = pd.DataFrame.from_dict(pgd_test_labels_dict)

# %%
reliability_diagram(
    fgsm_test_labels_dict['true_labels'], 
    fgsm_test_labels_dict['pred_labels'], 
    fgsm_test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='FGSM Adversarial test set performance (no conformal prediction)', figsize=(6, 6), dpi=100, return_fig=True
)

# %%
fgsm_empirical_coverage, fgsm_qhat, fgsm_prediction_sets = RAPS_conformal_prediction(
    cal_smx=cal_smx, cal_labels=cal_labels_dict['true_labels'],
    val_smx=fgsm_test_smx, val_labels=fgsm_test_labels_dict['true_labels'],
    lam_reg=0.01, k_reg=5, rand=False
)

def pred_set_sizes_hist(prediction_sets, cov):
    sizes = np.sum(prediction_sets, axis=1)
    bins = np.arange(np.min(sizes), np.max(sizes)+1)
    plt.hist(sizes, bins=bins, edgecolor='black')
    plt.xlabel('RAPS CP prediction set size')
    plt.ylabel('Frequency')
    plt.title(f'Unperturbed test set empirical coverage: {cov}')
    plt.show()

pred_set_sizes_hist(fgsm_prediction_sets, fgsm_empirical_coverage)

# %%

