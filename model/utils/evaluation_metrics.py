# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats

def evaluate_summary(y_pred, y_true, eval_method='avg'):
    """
    Calcula F-score, Kendall Tau y Spearman R entre dos resúmenes.
    y_pred, y_true: vectores 1D o 2D binarios/puntuaciones.
    """
    # Aplanar
    y_pred = np.ravel(y_pred)
    y_true = np.ravel(y_true)

    # Alinear longitudes si difieren
    if y_pred.shape[0] != y_true.shape[0]:
        min_len = min(y_pred.shape[0], y_true.shape[0])
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    if eval_method == 'avg':
        # F-score (binario, threshold 0.5)
        tp = np.sum((y_pred > 0.5) & (y_true > 0.5))
        p  = np.sum(y_pred > 0.5)
        r  = np.sum(y_true > 0.5)
        precision = tp / (p + 1e-8)
        recall    = tp / (r + 1e-8)
        f_score   = 2 * precision * recall / (precision + recall + 1e-8)

        # Kendall Tau
        kTau = stats.kendalltau(y_pred, y_true)[0]
        # Spearman R
        sRho = stats.spearmanr  (y_pred, y_true)[0]

        return f_score, kTau, sRho
    else:
        raise ValueError(f"Método de evaluación desconocido: {eval_method}")
