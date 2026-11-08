import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, roc_auc_score, confusion_matrix,
)

import pickle

def load_features(path):
    non_feature_cols = ["target_mean", "target_exceeds",  "target_severity", "event_id"]
    df               = pd.read_csv(path)
    feature_cols     = [c for c in df.columns if c not in non_feature_cols]
    return df, feature_cols

def add_severity_labels(df, moderate_threshold, severe_threshold):
    """Bin target_mean into normal/moderate/severe tiers."""
    df = df.copy()
    df["target_severity"] = np.select(
        [df["target_mean"] >= severe_threshold, df["target_mean"] >= moderate_threshold],
        [2, 1],
        default=0,
    )
    return df


def run_regression(train_df, val_df, feature_cols, target_col="target_mean"):
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val,   y_val   = val_df[feature_cols], val_df[target_col]

    models = {
        "linear": LinearRegression(),
        "lasso": Lasso(alpha=0.01),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=0),
        "gradient_boosting": GradientBoostingRegressor(random_state=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"[regression] {name}: MAE={mae:.3f}, R2={r2:.3f}")
        results[name] = {"model": model, "mae": mae, "r2": r2}

    return results


def run_classification(train_df, val_df, feature_cols, target_col="target_severity"):
    severity_labels = {0: "normal", 1: "moderate", 2: "severe"}
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val,   y_val   = val_df[feature_cols], val_df[target_col]

    present_classes = sorted(y_train.unique())
    target_names    = [severity_labels[c] for c in present_classes]

    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        if len(present_classes) > 2:
            auc = roc_auc_score(y_val, probs, multi_class="ovr", labels=present_classes)
        else:
            auc = roc_auc_score(y_val, probs[:, 1])

        print(f"\n[classification] {name} (AUC={auc:.3f})")
        print(classification_report(y_val, preds, labels=present_classes, target_names=target_names))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_val, preds, labels=present_classes))

        results[name] = {"model": model, "auc": auc}

    return results


def run_training(train_path="vil_tabular_features_train.csv",
                  test_path="vil_tabular_features_test.csv",
                  val_size=0.2,
                  moderate_quantile=0.90, severe_quantile=0.99):
    severity_labels        = {0: "normal", 1: "moderate", 2: "severe"}
    train_df, feature_cols = load_features(train_path)
    test_df,  _            = load_features(test_path)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Features: {feature_cols}")

    train_quantiles = train_df["target_mean"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"target_mean quantiles (train):\n{train_quantiles}")

    moderate_threshold = train_df["target_mean"].quantile(moderate_quantile)
    severe_threshold   = train_df["target_mean"].quantile(severe_quantile)
    print(f"Using moderate_threshold={moderate_threshold:.2f} (q={moderate_quantile}), "
          f"severe_threshold={severe_threshold:.2f} (q={severe_quantile})")

    train_df = add_severity_labels(train_df, moderate_threshold, severe_threshold)
    test_df  = add_severity_labels(test_df, moderate_threshold, severe_threshold)
    print(f"target_severity distribution (train):\n"
          f"{train_df['target_severity'].map(severity_labels).value_counts(normalize=True)}")

    train_sub, val_sub = train_test_split(train_df, test_size=val_size, random_state=0)

    print("\n===== Regression (val set) =====")
    reg_results = run_regression(train_sub, val_sub, feature_cols)

    print("\n===== Classification (val set) =====")
    clf_results = run_classification(train_sub, val_sub, feature_cols)

    return train_df, test_df, feature_cols, reg_results, clf_results


if __name__ == "__main__":
    
    train_df, test_df, feature_cols, reg_results, clf_results = run_training()
    train_quantiles = train_df["target_mean"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    X = train_df[feature_cols]

    pickle_save_object = {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "regression_results": reg_results,
        "classification_results": clf_results,
    }
    trained_tabular_models_picklefile = "trained_tabular_models.pickle"
    with open(trained_tabular_models_picklefile, "wb") as f:
        pickle.dump(pickle_save_object, f)