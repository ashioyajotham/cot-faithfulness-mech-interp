"""ROC curve and AUC plotting."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "Probe",
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a single ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
