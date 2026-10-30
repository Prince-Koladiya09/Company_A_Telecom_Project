
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

from .config import FIGURE_DIR


def evaluate_selected_model(results, model_name):
    """Evaluate the selected model using its validation-selected threshold."""

    y_test = results["y_test"]
    prediction = results["predictions"][model_name]

    y_pred = prediction["predicted"]
    y_prob = prediction["probability"]
    threshold = prediction["threshold"]

    print(f"Selected model: {model_name}")
    print(f"Decision threshold: {threshold:.4f}")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Class 0", "Class 1"],
            zero_division=0,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title(
        f"Confusion matrix — {model_name}\nThreshold={threshold:.3f}"
    )

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=axes[1],
    )
    axes[1].set_title(f"ROC curve — {model_name}")

    plt.tight_layout()

    path = FIGURE_DIR / "selected_model_evaluation.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return path
