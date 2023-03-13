# File: extract_softmax_scores.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/13/2023
# Description: Given a model extract softmax scores and labels from a dataset
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from easydict import EasyDict

def extract_softmax_scores(model, ds, n_classes=100):
    softmax_scores = np.empty((0, n_classes))
    y_star = np.empty(0)
    n = tf.data.experimental.cardinality(ds)
    for _, (x, y) in tqdm(enumerate(ds), total=n.numpy()):
        softmaxes = model(x)
        softmax_scores = np.vstack((softmax_scores, softmaxes.numpy()))
        y_star = np.concatenate((y_star, y.numpy()), axis=0)

    y_star = y_star.astype(int)
    y_hat = np.argmax(softmax_scores, axis=1).astype(int)
    y_conf = np.max(softmax_scores, axis=1)
    y_dict = EasyDict(true_labels=y_star, pred_labels=y_hat, confidences=y_conf)
    return softmax_scores, y_dict
