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
fgsm_test_ds = test_ds.unbatch().take(32).map(
    lambda x, y: (tf.squeeze(fast_gradient_method(
        logits_model, tf.expand_dims(x, axis=0), 
        eps=0.01, norm=np.inf, targeted=False
    )), y)
).batch(32)

# %%
# PGD attack
pgd_test_ds = test_ds.unbatch().take(32).map(
    lambda x, y: (tf.squeeze(projected_gradient_descent(
        logits_model, tf.cast(tf.expand_dims(x, axis=0), tf.float32), 
        eps=0.01, eps_iter=0.01, nb_iter=5,
        norm=np.inf, targeted=False
    )), y)
).batch(32)

# %%
fgsm_test_smx, fgsm_test_labels_dict = extract_softmax_scores(model, fgsm_test_ds)
pgd_test_smx, pgd_test_labels_dict = extract_softmax_scores(model, pgd_test_ds)

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

print(f'Saved plots from results block 3 to {FIG_DIR}')

# %%
def plot_random_sample(
    test_ds, fgsm_ds, pgd_ds, 
    test_preds, fgsm_preds, pgd_preds, 
    test_smxs, fgsm_smxs, pgd_smxs,
    skip=100, n_examples=10
):
    def un_preprocess(x):
        return tf.cast(tf.multiply(tf.cast(x, tf.float16), 255.), tf.int32)
    
    test_ds = test_ds.unbatch().skip(skip).take(n_examples).map(lambda x, y: (un_preprocess(x), y))
    fgsm_ds = fgsm_ds.unbatch().skip(skip).take(n_examples).map(lambda x, y: (un_preprocess(x), y))
    pgd_ds = pgd_ds.unbatch().skip(skip).take(n_examples).map(lambda x, y: (un_preprocess(x), y))
    test_p = test_preds[skip:(skip + n_examples), :]
    fgsm_p = fgsm_preds[skip:(skip + n_examples), :]
    pgd_p = pgd_preds[skip:(skip + n_examples), :]
    test_s = test_smxs[skip:(skip + n_examples), :]
    fgsm_s = fgsm_smxs[skip:(skip + n_examples), :]
    pgd_s = pgd_smxs[skip:(skip + n_examples), :]

    ds = tf.data.Dataset.zip((test_ds, fgsm_ds, pgd_ds))

    fig, ax = plt.subplots(nrows=n_examples, ncols=3, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': np.ones(n_examples, int).tolist()})
    for i, (og_sample, fgsm_sample, pgd_sample) in enumerate(ds):
        og_img, og_label = og_sample[0].numpy(), og_sample[1].numpy()
        fgsm_img = fgsm_sample[0].numpy()
        pgd_img = pgd_sample[0].numpy()

        og_lab_hr = human_readable_labels[og_label]

        hrl = np.array(human_readable_labels)

        og_idx = test_p[i, :]
        og_pred_set = hrl[og_idx]
        og_s = np.round(test_s[i, :][og_idx], 2)
        og_dict = dict(zip(og_pred_set, og_s))

        fgsm_idx = fgsm_p[i, :]
        fgsm_pred_set = hrl[fgsm_idx]
        fgsm_s = np.round(fgsm_s[i, :][fgsm_idx], 2)
        fgsm_dict = dict(zip(fgsm_pred_set, fgsm_s))

        pgd_idx = pgd_p[i, :]
        pgd_pred_set = hrl[pgd_idx]
        pgd_s = np.round(pgd_s[i, :][pgd_idx], 2)
        pgd_dict = dict(zip(pgd_pred_set, pgd_s))

        # Loop over each column in the row
        ax[i, 0].imshow(og_img)
        ax[i, 0].set_title(f'Unperturbed image, y = {og_lab_hr}\nCP pred set = {og_dict}')

        ax[i, 1].imshow(fgsm_img)
        ax[i, 1].set_title(f'FGSM attack\nCP pred set = {fgsm_dict}')

        ax[i, 2].imshow(pgd_img)
        ax[i, 2].set_title(f'PGD attack\nCP pred set = {pgd_dict}')

        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        fig.suptitle(f'CP behaviour on {n_examples} random test set examples')

    #fig.suptitle('Random test set images', fontsize=16)
    fig.savefig(f'{FIG_DIR}/test_set_images_skip{skip}_n{n_examples}.png', format='png', dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.show()

plot_random_sample(
    test_ds, fgsm_test_ds, pgd_test_ds, 
    prediction_sets, fgsm_prediction_sets, pgd_prediction_sets, 
    test_smx, fgsm_test_smx, pgd_test_smx,
    skip=0, n_examples=5
)
# %%
