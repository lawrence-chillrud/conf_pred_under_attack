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
from easydict import EasyDict
import matplotlib.pyplot as plt
from reliability_diagrams import *
from conformal_prediction import RAPS_conformal_prediction
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import tensorflow as tf
import seaborn as sns

# Hyperparameters
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'
FIG_DIR = f'/home/lawrence/conf_pred_under_attack/output/figures'

# Load and prep CIFAR-100
train_ds, cal_ds, val_ds, test_ds, human_readable_labels = download_cifar100_as_ds()
train_ds, cal_ds, val_ds, test_ds = prep_data(train_ds, cal_ds, val_ds, test_ds, batch_size=128)

# Load in model
model = init_model(freeze_base=False)
model.load_weights(f'{OUTPUT_DIR}/weights.h5')
print(model.summary())
model.compile(optimizer=Adam(lr=ALPHA), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
print("Obtaining result block 1... (4 progress bars should show here)")
for (ds_tag, ds) in zip(['Training', 'Validation', 'Calibration', 'Testing'], [train_ds, val_ds, cal_ds, test_ds]):
    loss, acc = model.evaluate(ds)
    print(f"{ds_tag} set performance: loss {loss}, accuracy {acc}")

# %%
print("Obtaining result block 2...")

# Extract softmax scores and labels from cal and test dataset
cal_smx, cal_labels_dict = extract_softmax_scores(model, cal_ds)
test_smx, test_labels_dict = extract_softmax_scores(model, test_ds)

# %%
plt.style.use("seaborn")
plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

test_calib_fig = reliability_diagram(
    test_labels_dict['true_labels'], 
    test_labels_dict['pred_labels'], 
    test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='Unperturbed test set calibration', figsize=(6, 6), dpi=100, return_fig=True
)
test_calib_fig.savefig(f'{FIG_DIR}/baseline_unperturbed_testset_performance.png', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)

# %%
empirical_coverage, qhat, prediction_sets = RAPS_conformal_prediction(
    cal_smx=cal_smx, cal_labels=cal_labels_dict['true_labels'],
    val_smx=test_smx, val_labels=test_labels_dict['true_labels'],
    lam_reg=0.01, k_reg=5, rand=False
)

def pred_set_sizes_hist(prediction_sets, cov, fname, title):
    sizes = np.sum(prediction_sets, axis=1)
    bins = np.arange(np.min(sizes), np.max(sizes)+1)
    plt.hist(sizes, bins=bins, edgecolor='black')
    plt.xlabel('RAPS CP prediction set size')
    plt.ylabel('Frequency')
    plt.title(f'{title} (overall empirical coverage: {cov})')
    plt.savefig(f'{FIG_DIR}/{fname}', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.show()

pred_set_sizes_hist(
    prediction_sets, empirical_coverage, 
    fname='baseline_unperturbed_testset_prediction_set_size_dist.png', 
    title='RAPS CP prediction set size distribution in the unperturbed test set'
)

# %%
def pred_set_by_conf_scatter(pred_sets, conf, cov, title, fname):
    pred_set_len = np.sum(pred_sets, axis=1)
    bp_df = pd.DataFrame(EasyDict(pred_set_len=pred_set_len, conf=conf))
    boxprops = dict(linewidth=2, color='tab:blue', edgecolor='black')
    sns.boxplot(x='pred_set_len', y='conf', data=bp_df, boxprops=boxprops)
    sns.stripplot(x='pred_set_len', y='conf', data=bp_df, color='tab:orange', edgecolor='black', alpha=0.5, size=4, jitter=0.3)
    plt.title(f'{title} (overall empirical coverage: {cov})')
    plt.xlabel("Prediction set size")
    plt.ylabel("Max softmax score (confidence) in prediction set")
    plt.savefig(f'{FIG_DIR}/{fname}', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.show()

pred_set_by_conf_scatter(
    pred_sets=prediction_sets, conf=test_labels_dict['confidences'], cov=empirical_coverage,
    title='RAPS CP in the unperturbed test set',
    fname='baseline_unperturbed_testset_predset_by_conf_scatter.png'
)
print(f'Saved plots from results block 2 to {FIG_DIR}')

# %%
# Construct adversarial attacks
print("Obtaining result block 3 (this may take a while since we make the adversarial attacks here)...")

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
pgd_test_ds = test_ds.unbatch().take(BATCH_SIZE*3).map(
    lambda x, y: (tf.squeeze(projected_gradient_descent(
        logits_model, tf.expand_dims(x, axis=0), 
        eps=0.01, eps_iter=0.01, nb_iter=20,
        norm=np.inf, targeted=False
    )), y)
).batch(BATCH_SIZE)

# %%
fgsm_test_smx, fgsm_test_labels_dict = extract_softmax_scores(model, fgsm_test_ds)
#fgsm_test_df = pd.DataFrame.from_dict(fgsm_test_labels_dict)

# %%
pgd_test_smx, pgd_test_labels_dict = extract_softmax_scores(model, pgd_test_ds)
#pgd_test_df = pd.DataFrame.from_dict(pgd_test_labels_dict)

# %%
fgsm_test_calib_fig = reliability_diagram(
    fgsm_test_labels_dict['true_labels'], 
    fgsm_test_labels_dict['pred_labels'], 
    fgsm_test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='FGSM adversarial test set calibration', figsize=(6, 6), dpi=100, return_fig=True
)
fgsm_test_calib_fig.savefig(f'{FIG_DIR}/fgsm_testset_performance.png', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)

pgd_test_calib_fig = reliability_diagram(
    pgd_test_labels_dict['true_labels'], 
    pgd_test_labels_dict['pred_labels'], 
    pgd_test_labels_dict['confidences'],
    num_bins=10, draw_ece=True,
    draw_bin_importance="alpha", draw_averages=True,
    title='PGD adversarial test set calibration', figsize=(6, 6), dpi=100, return_fig=True
)
pgd_test_calib_fig.savefig(f'{FIG_DIR}/pgd_testset_performance.png', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)

# %%
fgsm_empirical_coverage, fgsm_qhat, fgsm_prediction_sets = RAPS_conformal_prediction(
    cal_smx=cal_smx, cal_labels=cal_labels_dict['true_labels'],
    val_smx=fgsm_test_smx, val_labels=fgsm_test_labels_dict['true_labels'],
    lam_reg=0.01, k_reg=5, rand=False
)

pgd_empirical_coverage, pgd_qhat, pgd_prediction_sets = RAPS_conformal_prediction(
    cal_smx=cal_smx, cal_labels=cal_labels_dict['true_labels'],
    val_smx=pgd_test_smx, val_labels=pgd_test_labels_dict['true_labels'],
    lam_reg=0.01, k_reg=5, rand=False
)

pred_set_sizes_hist(
    fgsm_prediction_sets, fgsm_empirical_coverage, 
    fname='fgsm_testset_prediction_set_size_dist.png', 
    title='RAPS CP prediction set size distribution in the FGSM adversarial test set'
)

pred_set_sizes_hist(
    pgd_prediction_sets, pgd_empirical_coverage, 
    fname='pgd_testset_prediction_set_size_dist.png', 
    title='RAPS CP prediction set size distribution in the PGD adversarial test set'
)

pred_set_by_conf_scatter(
    pred_sets=fgsm_prediction_sets, conf=fgsm_test_labels_dict['confidences'], cov=fgsm_empirical_coverage,
    title='RAPS CP in the FGSM adversarial test set',
    fname='fgsm_testset_predset_by_conf_scatter.png'
)

pred_set_by_conf_scatter(
    pred_sets=pgd_prediction_sets, conf=pgd_test_labels_dict['confidences'], cov=pgd_empirical_coverage,
    title='RAPS CP in the PGD adversarial test set',
    fname='pgd_testset_predset_by_conf_scatter.png'
)

print(f'Saved plots from results block 3 to {FIG_DIR}')

def cp_by_adversarial_scatter(cp_pred_sets, adv_pred_sets, cp_cov, adv_cov, title, fname, xlab='Unperturbed CP prediction set size', ylab='Adversarial CP prediction set size'):
    cp_pred_set_len = np.sum(cp_pred_sets, axis=1)
    adv_pred_set_len = np.sum(adv_pred_sets, axis=1)
    adv_max_val = np.max(adv_pred_set_len) 
    cp_max_val = np.max(cp_pred_set_len)
    plt.plot([0, cp_max_val], [0, adv_max_val], linestyle='--', color='black', alpha=0.5, zorder=0)
    plt.scatter(x=cp_pred_set_len, y=adv_pred_set_len, color='tab:blue', edgecolor='black', alpha=0.05, zorder=1)
    plt.xticks(np.arange(1, cp_max_val + 1))
    plt.yticks(np.arange(1, adv_max_val + 1))
    plt.title(f'{title}\n(Coverage {cp_cov} vs. {adv_cov})')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(f'{FIG_DIR}/{fname}', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.show()

def cp_by_adversarial_bp(cp_pred_sets, adv_pred_sets, cp_cov, adv_cov, title, fname, xlab='Unperturbed CP prediction set size', ylab='Adversarial CP prediction set size'):
    cp_pred_set_len = np.sum(cp_pred_sets, axis=1)
    adv_pred_set_len = np.sum(adv_pred_sets, axis=1)
    adv_max_val = np.max(adv_pred_set_len) 
    cp_max_val = np.max(cp_pred_set_len)
    bp_df = pd.DataFrame(EasyDict(cp_pred_set_len=cp_pred_set_len, adv_pred_set_len=adv_pred_set_len))
    boxprops = dict(linewidth=2, color='tab:blue', edgecolor='black')
    sns.set_style('whitegrid')
    sns.boxplot(x='cp_pred_set_len', y='adv_pred_set_len', data=bp_df, boxprops=boxprops)
    sns.stripplot(x='cp_pred_set_len', y='adv_pred_set_len', data=bp_df, color='tab:orange', edgecolor='black', alpha=0.5, size=4, jitter=0.3)
    plt.plot([0, cp_max_val], [0, adv_max_val], linestyle='--', color='black', alpha=0.5, zorder=0)
    plt.xticks(np.arange(1, cp_max_val + 1))
    plt.yticks(np.arange(1, adv_max_val + 1))
    plt.title(f'{title}\n(Coverage {cp_cov} vs. {adv_cov})')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(f'{FIG_DIR}/{fname}', format='png', dpi=100, bbox_inches="tight", pad_inches=0.2)
    plt.show()

cp_by_adversarial_scatter(
    cp_pred_sets=prediction_sets, adv_pred_sets=fgsm_prediction_sets, 
    cp_cov=empirical_coverage, adv_cov=fgsm_empirical_coverage,
    title='Unperturbed vs. FGSM adversarial CP prediction set sizes',
    xlab='Unperturbed CP prediction set size', ylab='FGSM adversarial CP prediction set size',
    fname='cp_vs_fgsm_set_size.png'
)

cp_by_adversarial_scatter(
    cp_pred_sets=prediction_sets, adv_pred_sets=pgd_prediction_sets, 
    cp_cov=empirical_coverage, adv_cov=pgd_empirical_coverage,
    title='Unperturbed vs. PGD adversarial CP prediction set sizes',
    xlab='Unperturbed CP prediction set size', ylab='PGD adversarial CP prediction set size',
    fname='cp_vs_pgd_set_size.png'
)

cp_by_adversarial_scatter(
    cp_pred_sets=fgsm_prediction_sets, adv_pred_sets=pgd_prediction_sets, 
    cp_cov=fgsm_empirical_coverage, adv_cov=pgd_empirical_coverage,
    title='FGSM vs. PGD adversarial CP prediction set sizes',
    xlab='FGSM adversarial CP prediction set size', ylab='PGD adversarial CP prediction set size',
    fname='fgsm_vs_pgd_set_size.png'
)

cp_by_adversarial_bp(
    cp_pred_sets=prediction_sets, adv_pred_sets=fgsm_prediction_sets, 
    cp_cov=empirical_coverage, adv_cov=fgsm_empirical_coverage,
    title='Unperturbed vs. FGSM adversarial CP prediction set sizes',
    xlab='Unperturbed CP prediction set size', ylab='FGSM adversarial CP prediction set size',
    fname='cp_vs_fgsm_set_size_bp.png'
)

cp_by_adversarial_bp(
    cp_pred_sets=prediction_sets, adv_pred_sets=pgd_prediction_sets, 
    cp_cov=empirical_coverage, adv_cov=pgd_empirical_coverage,
    title='Unperturbed vs. PGD adversarial CP prediction set sizes',
    xlab='Unperturbed CP prediction set size', ylab='PGD adversarial CP prediction set size',
    fname='cp_vs_pgd_set_size_bp.png'
)

cp_by_adversarial_bp(
    cp_pred_sets=fgsm_prediction_sets, adv_pred_sets=pgd_prediction_sets, 
    cp_cov=fgsm_empirical_coverage, adv_cov=pgd_empirical_coverage,
    title='FGSM vs. PGD adversarial CP prediction set sizes',
    xlab='FGSM adversarial CP prediction set size', ylab='PGD adversarial CP prediction set size',
    fname='fgsm_vs_pgd_set_size_bp.png'
)

# %%

