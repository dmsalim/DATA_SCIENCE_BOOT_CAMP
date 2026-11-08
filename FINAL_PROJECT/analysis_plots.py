import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import os
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve



def plot_regression_diagnostics(reg_results, val_df, feature_cols, target_col="target_mean", savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]

    n_models = len(reg_results)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))

    for i, (name, result) in enumerate(reg_results.items()):
        preds = result["model"].predict(X_val)
        residuals = y_val - preds

        # Top row: predicted vs actual
        ax = axes[0, i]
        ax.scatter(y_val, preds, alpha=0.2, s=5)
        lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1)  # perfect-prediction line
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nMAE={result['mae']:.2f}, R2={result['r2']:.3f}")

        # Bottom row: residuals vs predicted
        ax = axes[1, i]
        ax.scatter(preds, residuals, alpha=0.2, s=5)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "regression_diagnostics.pdf"), bbox_inches="tight")

    return fig

def plot_classification_diagnostics(clf_results, val_df, feature_cols,
                                     target_col="target_severity", class_names=None, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]

    if class_names is None:
        class_names = sorted(y_val.unique())

    n_models = len(clf_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, clf_results.items()):
        preds = result["model"].predict(X_val)
        cm = confusion_matrix(y_val, preds, labels=sorted(y_val.unique()))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAUC={result['auc']:.3f}")

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "classification_confusion_matrices.pdf"), bbox_inches="tight")

    return fig

def plot_feature_importance(model, feature_cols, title, ax=None):
    if not hasattr(model, "feature_importances_"):
        print(f"{title}: no feature_importances_ (not a tree-based model)")
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(title)
    return ax

def plot_all_feature_importances(reg_results, clf_results, feature_cols, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_feature_importance(
        reg_results["gradient_boosting"]["model"], feature_cols,
        "Regression (gradient boosting)", ax=axes[0],
    )
    plot_feature_importance(
        clf_results["gradient_boosting"]["model"], feature_cols,
        "Classification (gradient boosting)", ax=axes[1],
    )
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "feature_importance_comparison.pdf"), bbox_inches="tight")
    return fig

def plot_calibration(clf_results, val_df, feature_cols, target_col="target_severity",
                      positive_class=2, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    """Checks whether predicted probabilities for the 'severe' class are well-calibrated."""
    severity_labels = {0: "normal", 1: "moderate", 2: "severe"}
    y_val = (val_df[target_col] == positive_class).astype(int)
    X_val = val_df[feature_cols]

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, result in clf_results.items():
        probs = result["model"].predict_proba(X_val)
        # column index for the positive_class label
        class_idx = list(result["model"].classes_).index(positive_class)
        prob_pos = probs[:, class_idx]

        frac_pos, mean_pred = calibration_curve(y_val, prob_pos, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual positives")
    ax.set_title(f"Calibration — class {severity_labels[positive_class]}")
    ax.legend()
    fig.savefig(os.path.join(savepath, "calibration_severe.pdf"), bbox_inches="tight")
    return fig

if __name__ == "__main__":
    trained_tabular_models_picklefile = "trained_tabular_models.pickle"
    with open(trained_tabular_models_picklefile, 'rb') as f:
        train_tabular_pickle_save_object = pickle.load(f)

    train_df = train_tabular_pickle_save_object["train_df"]
    test_df = train_tabular_pickle_save_object["test_df"]
    feature_cols = train_tabular_pickle_save_object["feature_cols"]
    reg_results = train_tabular_pickle_save_object["regression_results"]
    clf_results = train_tabular_pickle_save_object["classification_results"]

    plot_regression_diagnostics(reg_results, test_df, feature_cols)
    plot_classification_diagnostics(clf_results, test_df, feature_cols,
                                 class_names=["normal", "moderate", "severe"])
    plot_all_feature_importances(reg_results, clf_results, feature_cols)
    plot_calibration(clf_results, test_df, feature_cols)