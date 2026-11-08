import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from train_tabular_models import load_features, add_severity_labels, SEVERITY_LABELS


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


if __name__ == "__main__":
    train_df = run_anomaly_detection()