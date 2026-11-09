import os
import matplotlib.pyplot    as plt
import pandas               as pd
from   sklearn.ensemble     import IsolationForest
from   sklearn.svm          import OneClassSVM
from   sklearn.neighbors    import LocalOutlierFactor

from   train_tabular_models import load_features, add_severity_labels
from   sevir_utils          import SEVIRVILDataset



def run_anomaly_detection(train_path="vil_tabular_features_train.csv",
                           contamination=0.05, n_neighbors=20):
    train_df, feature_cols = load_features(train_path)
    X = train_df[feature_cols]

    print(f"Running anomaly detection on {train_df.shape[0]} rows, {len(feature_cols)} features")

    # Isolation Forest — isolates points that are "easy to separate" via random splits
    iso = IsolationForest(contamination=contamination, random_state=0)
    train_df["anomaly_iso"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal

    # One-Class SVM — learns a boundary around "normal" data in feature space
    ocsvm = OneClassSVM(nu=contamination)
    train_df["anomaly_svm"] = ocsvm.fit_predict(X)

    # Local Outlier Factor — flags points whose local density is much lower than their neighbors
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    train_df["anomaly_lof"] = lof.fit_predict(X)

    for col in ["anomaly_iso", "anomaly_svm", "anomaly_lof"]:
        print(f"{col}:\n{train_df[col].value_counts()}\n")

    return train_df

def add_severity_to_anomaly_df(train_df, moderate_quantile=0.90, severe_quantile=0.99):
    """Bins target_mean into the same normal/moderate/severe tiers used by the
    classification models, so anomaly flags can be checked against them."""
    moderate_threshold = train_df["target_mean"].quantile(moderate_quantile)
    severe_threshold = train_df["target_mean"].quantile(severe_quantile)
    train_df = add_severity_labels(train_df, moderate_threshold, severe_threshold)
    return train_df, moderate_threshold, severe_threshold


def compute_crosstabs(train_df):
    """Returns one cross-tabulation DataFrame per anomaly method, each showing
    how its -1/1 flag distributes across the three severity tiers."""
    severity_labels = {0: "normal", 1: "moderate", 2: "severe"}
    severity_named  = train_df["target_severity"].map(severity_labels)
    crosstabs = {}
    for col in ["anomaly_iso", "anomaly_svm", "anomaly_lof"]:
        ct = pd.crosstab(train_df[col], severity_named, margins=True)
        ct = ct.rename(index={-1: "anomaly", 1: "normal", "All": "Total"})
        crosstabs[col] = ct
    return crosstabs


def plot_crosstabs(crosstabs, severity_tier_totals, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    """Bar chart showing, for each detector, what FRACTION of each severity
    tier's windows were flagged as anomalous -- normalizes for the fact that
    normal/moderate/severe are highly imbalanced in the underlying data."""
    fig, axes = plt.subplots(1, len(crosstabs), figsize=(5 * len(crosstabs), 4.5))
    if len(crosstabs) == 1:
        axes = [axes]

    tier_order = ["normal", "moderate", "severe"]

    for ax, (name, ct) in zip(axes, crosstabs.items()):
        if "anomaly" not in ct.index:
            ax.set_title(f"{name}\n(no anomalies flagged)")
            continue
        present_tiers = [c for c in tier_order if c in ct.columns]
        flagged_counts = ct.loc["anomaly", present_tiers]
        rates = flagged_counts / pd.Series({t: severity_tier_totals[t] for t in present_tiers})

        rates.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452", "#c44e52"])
        ax.set_title(name.replace("anomaly_", "").upper())
        ax.set_ylabel("Fraction of tier flagged as anomalous")
        ax.set_xlabel("Quantile-derived severity tier")
        ax.set_ylim(0, 1)

    fig.suptitle("Anomaly flag rate by severity tier, by detection method")
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, "anomaly_severity_crosstab.pdf"), bbox_inches="tight")
    return fig

def generate_crosstab_latex(crosstabs, severity_tier_totals, savepath=os.path.join(os.getcwd(), "ANALYSIS_PLOTS")):
    """Builds one combined LaTeX table showing each detector's flagged-anomaly
    counts AND flag rates (count / tier total) broken down by quantile-derived
    severity tier -- the rate accounts for the heavy imbalance across tiers."""
    tier_order = ["normal", "moderate", "severe"]
    rows = []
    for name, ct in crosstabs.items():
        display_name = name.replace("anomaly_", "").upper()
        if "anomaly" not in ct.index:
            rows.append(f"{display_name} & \\multicolumn{{3}}{{c}}{{no anomalies flagged}} \\\\")
            continue
        cells = []
        for t in tier_order:
            count = ct.loc["anomaly", t] if t in ct.columns else 0
            total = severity_tier_totals.get(t, 0)
            rate = count / total if total > 0 else float("nan")
            cells.append(f"{count} ({rate:.1%})")
        rows.append(f"{display_name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Detector & Normal & Moderate & Severe \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Count (and rate, i.e.\\ count divided by the total number of windows "
        "in that tier) of windows flagged as anomalous by each unsupervised method, "
        "broken down by quantile-derived severity tier. The rate accounts for the "
        "substantial imbalance in tier sizes (normal/moderate/severe windows occur at "
        "approximately 90\\%/9\\%/1\\% of the training set respectively) and is the "
        "quantity directly comparable to Figure~\\ref{fig:anomaly_crosstab}.}\n"
        "\\label{tab:anomaly_crosstab}\n"
        "\\end{table}\n"
    )

    with open(os.path.join(savepath, "anomaly_crosstab_table.tex"), "w") as f:
        f.write(table)

    print(table)
    return table

def compute_agreement_stats(train_df):
    """Fraction of each detector's flagged anomalies that fall in the severe tier,
    and the fraction of true severe windows that each detector actually catches --
    the two numbers that matter most for the discussion paragraph."""
    stats = {}
    n_severe_total = (train_df["target_severity"] == 2).sum()

    for col in ["anomaly_iso", "anomaly_svm", "anomaly_lof"]:
        flagged = train_df[train_df[col] == -1]
        n_flagged = len(flagged)
        n_flagged_severe = (flagged["target_severity"] == 2).sum()

        precision_vs_severe = n_flagged_severe / n_flagged if n_flagged > 0 else float("nan")
        recall_vs_severe = n_flagged_severe / n_severe_total if n_severe_total > 0 else float("nan")

        stats[col] = {
            "n_flagged": n_flagged,
            "n_flagged_severe": n_flagged_severe,
            "precision_vs_severe": precision_vs_severe,
            "recall_vs_severe": recall_vs_severe,
        }
        print(f"{col}: {n_flagged} flagged, {n_flagged_severe} of those are 'severe' "
              f"({precision_vs_severe:.1%} precision, {recall_vs_severe:.1%} recall vs. severe tier)")

    return stats


if __name__ == "__main__":
    severity_labels = {0: "normal", 1: "moderate", 2: "severe"}
    savepath        = os.path.join(os.getcwd(), "ANALYSIS_PLOTS")
    train_df        = run_anomaly_detection()
    train_df, moderate_threshold, severe_threshold = add_severity_to_anomaly_df(train_df)

    severity_tier_totals = train_df["target_severity"].map(severity_labels).value_counts().to_dict()

    crosstabs = compute_crosstabs(train_df)
    for name, ct in crosstabs.items():
        print(f"\n{name}:\n{ct}")

    plot_crosstabs(crosstabs, severity_tier_totals, savepath=savepath)
    generate_crosstab_latex(crosstabs, severity_tier_totals, savepath=savepath)
    agreement_stats = compute_agreement_stats(train_df)