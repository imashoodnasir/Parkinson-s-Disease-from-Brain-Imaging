from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score
)

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    """Compute Acc/F1/AUC/Sensitivity/Specificity for binary classification (PD=1, HC=0)."""
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thresh).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spe  = tn / (tn + fp + 1e-12)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    return {
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spe),
        "precision": float(prec),
        "recall": float(rec),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
