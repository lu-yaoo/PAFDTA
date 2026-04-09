# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple
from sklearn.metrics import r2_score, average_precision_score
# Requirement:pip install scikit-learn

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))

def aupr_from_scores(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float:
    """
    y_true_binary: 0/1
    y_scores:  ( " ")
    """
    y_true_binary = np.asarray(y_true_binary).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    return float(average_precision_score(y_true_binary, y_scores))

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, block: int = 1024) -> float:
    """
      Harrell's C(  DTA   CI  , y_true ;y_pred 0.5).
      O(N^2) Python  .
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.shape[0]
    if n < 2:
        return 1.0

    total_pairs = 0
    correct = 0.0

    for i in range(0, n, block):
        i_end = min(i + block, n)
        yt_i = y_true[i:i_end][:, None]  # [bi, 1]
        yp_i = y_pred[i:i_end][:, None]
        for j in range(i + 1, n, block):
            j_end = min(j + block, n)
            yt_j = y_true[j:j_end][None, :]  # [1, bj]
            yp_j = y_pred[j:j_end][None, :]

            dy = yt_i - yt_j          # >0   i>j
            dp = yp_i - yp_j

            valid = (dy != 0.0)
            if not np.any(valid):
                continue

            conc = (dp * dy) > 0.0
            ties_p = (dp == 0.0)

            total_pairs += np.sum(valid)
            #   valid
            correct += np.sum(conc & valid) + 0.5 * np.sum(ties_p & valid)

    if total_pairs == 0:
        return 1.0
    return float(correct / total_pairs)

def pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    #  :  Pearson / Spearman,
    from scipy.stats import pearsonr, spearmanr
    pr = float(pearsonr(y_true, y_pred)[0])
    sr = float(spearmanr(y_true, y_pred)[0])
    return pr, sr
"""Evaluation metrics used for drug-target affinity prediction."""

