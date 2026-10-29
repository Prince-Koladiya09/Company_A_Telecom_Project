import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)

from .config import FIGURE_DIR

def evaluate_selected_model(results, model_name):
    y_test = results["y_test"]
    prediction = results["predictions"][model_name]
    y_pred = prediction["predicted"]
    y_prob = prediction["probability"]

    print(f"Selected model: {model_name}")
    print(classification_report(
        y_test, y_pred, target_names=["Class 0", "Class 1"], zero_division=0
    ))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"Confusion matrix — {model_name}")
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    axes[1].set_title(f"ROC curve — {model_name}")
    plt.tight_layout()
    path = FIGURE_DIR / "selected_model_evaluation.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    return path
