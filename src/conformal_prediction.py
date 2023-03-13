# File: conformal_prediction.py
# Author: Adapted by Lawrence Chillrud <chili@u.northwestern.edu>, originally written by Anastasios Angelopoulos <angelopoulos@berkeley.edu>
# Date: 03/13/2023
# Description: Implement RAPS conformal prediction as detailed by Angelopoulos et al.
# Link to original code: <https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb>
import numpy as np

def RAPS_conformal_prediction(cal_smx, cal_labels, val_smx, val_labels, n_classes=100, n_cal=1000, alpha=0.1, lam_reg=0.01, k_reg=5, disallow_zero_sets=False, rand=True):
    reg_vec = np.array(k_reg*[0,] + (n_classes-k_reg)*[lam_reg,])[None,:]
    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
    empirical_coverage = prediction_sets[np.arange(n_val),val_labels].mean()
    print(f"The empirical coverage is: {empirical_coverage}")
    print(f"The quantile is: {qhat}")