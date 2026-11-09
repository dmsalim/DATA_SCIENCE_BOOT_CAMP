import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import os
import pickle

from sklearn.metrics     import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sevir_utils         import SEVIRVILDataset

# --- Windows per event and total windows (input_len=13, output_len=12, stride=12) ---
def n_windows_per_event(T=49, input_len=13, output_len=12, stride=12):
    return len(range(0, T - input_len - output_len + 1, stride))

def generate_regression_latex_table(reg_results, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    """Builds a LaTeX table (as a string) from the regression results dict and
    saves it to a .tex file for direct \\input{} inclusion in the report."""
    rows = []
    for name, result in reg_results.items():
        display_name = name.replace("_", " ").title()
        rows.append(f"{display_name} & {result['mae']:.3f} & {result['r2']:.3f} \\\\")

    table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Model & MAE & $R^2$ \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Regression performance predicting \\texttt{target\\_mean} on the held-out validation split.}\n"
        "\\label{tab:regression_results}\n"
        "\\end{table}\n"
    )

    with open(os.path.join(savepath, "results_regression_table.tex"), "w") as f:
        f.write(table)

    print(table)
    return table


def generate_classification_latex_table(clf_results, val_df, feature_cols,
                                          target_col="target_severity",
                                          class_names=("normal", "moderate", "severe"),
                                          savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    """Builds a LaTeX table of per-class precision/recall/F1 plus AUC for each
    classifier, from the classification results dict and validation data."""
    from sklearn.metrics import classification_report

    y_val = val_df[target_col]
    X_val = val_df[feature_cols]
    present_classes = sorted(y_val.unique())
    target_names = [class_names[c] for c in present_classes]

    rows = []
    for name, result in clf_results.items():
        preds = result["model"].predict(X_val)
        report = classification_report(
            y_val, preds, labels=present_classes, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        display_name = name.replace("_", " ").title()
        rows.append(
            f"\\multirow{{{len(target_names)}}}{{*}}{{{display_name}}}"
        )
        for i, cls in enumerate(target_names):
            prec = report[cls]["precision"]
            rec = report[cls]["recall"]
            f1 = report[cls]["f1-score"]
            support = int(report[cls]["support"])
            if i == 0:
                rows[-1] += f" & {cls} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {support} \\\\"
            else:
                rows.append(f" & {cls} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {support} \\\\")
        rows.append(f"& \\textbf{{AUC}} & \\multicolumn{{4}}{{l}}{{{result['auc']:.3f}}} \\\\")
        rows.append("\\midrule")

    if rows and rows[-1] == "\\midrule":
        rows.pop()

    table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        "Model & Class & Precision & Recall & F1 & Support \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Classification performance predicting \\texttt{target\\_severity} "
        "(3-class, macro one-vs-rest AUC reported per model) on the held-out validation split.}\n"
        "\\label{tab:classification_results}\n"
        "\\end{table}\n"
    )

    with open(os.path.join(savepath, "results_classification_table.tex"), "w") as f:
        f.write(table)

    print(table)
    return table

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
    ##################################
    ### --- PRINT DATA METRICS --- ###
    ##################################

    # --- Catalog-level counts ---
    catalog = pd.read_csv("CATALOG.csv", parse_dates=["time_utc"], low_memory=False)
    print("Full catalog rows:", catalog.shape[0])
    print("Catalog rows by modality:\n", catalog["img_type"].value_counts())

    vil_catalog = catalog[catalog["img_type"] == "vil"]
    print("VIL-only catalog rows:", len(vil_catalog))

    train_catalog = pd.read_csv("vil_train_catalog.csv", parse_dates=["time_utc"])
    test_catalog = pd.read_csv("vil_test_catalog.csv", parse_dates=["time_utc"])
    print("Train events:", len(train_catalog))
    print("Test events:", len(test_catalog))

    # --- HDF5 file count for the VIL modality (run this on the cluster, in the data/vil directory) ---
    # find data/vil -name "*.h5" | wc -l

    windows_per_event = n_windows_per_event()
    print("Windows per event:", windows_per_event)
    print("Total train windows (approx):", len(train_catalog) * windows_per_event)
    print("Total test windows (approx):", len(test_catalog) * windows_per_event)

    # --- Severity threshold values, from your tabular feature CSV ---
    tabular_train = pd.read_csv("vil_tabular_features_train.csv")
    print("target_mean quantiles:\n", tabular_train["target_mean"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    moderate_threshold = tabular_train["target_mean"].quantile(0.90)
    severe_threshold = tabular_train["target_mean"].quantile(0.99)
    print("moderate_threshold:", moderate_threshold)
    print("severe_threshold:", severe_threshold)

    print("Actual windows in tabular_train:", len(tabular_train))

    # or, if using the PyTorch dataset directly:
    train_dataset = SEVIRVILDataset(train_catalog, data_dir="data")
    print("Actual windows in SEVIRVILDataset:", len(train_dataset))

    ##########################################
    ### --- TRAINED RESULTS START HERE --- ###
    ##########################################

    trained_tabular_models_picklefile = "trained_tabular_models.pickle"
    with open(trained_tabular_models_picklefile, 'rb') as f:
        train_tabular_pickle_save_object = pickle.load(f)

    train_df     = train_tabular_pickle_save_object["train_df"]
    test_df      = train_tabular_pickle_save_object["test_df"]
    feature_cols = train_tabular_pickle_save_object["feature_cols"]
    reg_results  = train_tabular_pickle_save_object["regression_results"]
    clf_results  = train_tabular_pickle_save_object["classification_results"]

    plot_regression_diagnostics(reg_results, test_df, feature_cols)
    plot_classification_diagnostics(clf_results, test_df, feature_cols,
                                 class_names=["normal", "moderate", "severe"])
    plot_all_feature_importances(reg_results, clf_results, feature_cols)
    plot_calibration(clf_results, test_df, feature_cols)

    generate_regression_latex_table(reg_results)
    generate_classification_latex_table(clf_results, test_df, feature_cols) 

    reg_importances = pd.Series(
        reg_results["gradient_boosting"]["model"].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    clf_importances = pd.Series(
        clf_results["gradient_boosting"]["model"].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("Regression (gradient boosting) feature importances:")
    print(reg_importances)
    print("\nClassification (gradient boosting) feature importances:")
    print(clf_importances)

    # Quick agreement check: how much do the top-3 features overlap?
    top3_reg = set(reg_importances.head(3).index)
    top3_clf = set(clf_importances.head(3).index)
    print("\nTop-3 overlap:", top3_reg & top3_clf)