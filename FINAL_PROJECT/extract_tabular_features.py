import os
import argparse
import numpy as np
import pandas as pd

from sevir_utils import load_event, make_windows


def extract_features(x, y, predict_frame=11, rain_threshold=74):
    """
    x: input window, shape (H, W, input_len) - context frames
    y: target window, shape (H, W, output_len) - future frames
    predict_frame: which future frame to predict (0 = next frame, 11 = 60 min ahead)
    rain_threshold: VIL value above which a pixel counts as "exceeding" for classification
    """
    last_frame = x[:, :, -1]
    prev_frame = x[:, :, -2]

    features = {
        "mean_intensity_last"        : last_frame.mean(),
        "max_intensity_last"         : last_frame.max(),
        "std_intensity_last"         : last_frame.std(),
        "rain_area_last"             : (last_frame > 0).sum(),
        "mean_intensity_trend"       : last_frame.mean() - prev_frame.mean(),
        "max_intensity_trend"        : last_frame.max() - prev_frame.max(),
        "mean_intensity_full_window" : x.mean(),
        "max_intensity_full_window"  : x.max(),
    }

    target_frame   = y[:, :, predict_frame]
    target_mean    = target_frame.mean()
    target_exceeds = int((target_frame > rain_threshold).any())

    return features, target_mean, target_exceeds


def build_tabular_dataset(catalog, data_dir, input_len=13, output_len=12,
                           stride=12, predict_frame=11, rain_threshold=74, limit=None):
    rows = []
    n_events = len(catalog) if limit is None else min(limit, len(catalog))

    for i, (idx, row) in enumerate(catalog.iterrows()): #each row is one SEVIR event
        if limit is not None and i >= limit:
            break

        seq = load_event(data_dir, row["file_name"], row["file_index"])

        for x, y in make_windows(seq, input_len, output_len, stride):
            feats, target_mean, target_exceeds = extract_features(x, y, predict_frame, rain_threshold)
            feats["target_mean"]               = target_mean
            feats["target_exceeds"]            = target_exceeds
            feats["event_id"]                  = row["id"]
            rows.append(feats)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{n_events} events, {len(rows)} windows so far")

    return pd.DataFrame(rows)

def run_extraction(catalogs, outputs, data_dir=None,
                    predict_frame=11, rain_threshold=74, limit=None):
    """
    catalogs: list of catalog CSV paths, e.g. ["vil_train_catalog.csv", "vil_test_catalog.csv"]
    outputs: matching list of output CSV paths, one per catalog, in the same order
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    if len(catalogs) != len(outputs):
        raise ValueError(
            f"Got {len(catalogs)} catalogs but {len(outputs)} outputs — "
            "these must be passed in matching pairs."
        )

    results = {}
    for catalog_path, output_path in zip(catalogs, outputs):
        print(f"\n=== Processing {catalog_path} ===")
        catalog = pd.read_csv(catalog_path, parse_dates=["time_utc"])
        print(f"Loaded catalog: {catalog.shape}")

        tabular_df = build_tabular_dataset(catalog, data_dir,
                                           predict_frame  = predict_frame,
                                           rain_threshold = rain_threshold,
                                           limit          = limit,
        )
        print(f"Final tabular shape: {tabular_df.shape}")
        print(tabular_df.describe())
        print(f"target_exceeds distribution:\n{tabular_df['target_exceeds'].value_counts(normalize=True)}")

        tabular_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

        results[output_path] = tabular_df

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogs", nargs="+", required=True,
                         help="One or more catalog CSV paths, e.g. vil_train_catalog.csv vil_test_catalog.csv")
    parser.add_argument("--outputs", nargs="+", required=True,
                         help="Matching output CSV paths, one per catalog, in the same order")
    parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data"),
                         help="Path to the SEVIR data directory containing HDF5 files")
    parser.add_argument("--predict_frame", type=int, default=11)
    parser.add_argument("--rain_threshold", type=float, default=74,
                         help="VIL value above which a pixel counts as exceeding, for the classification target")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process first N events per split (for quick testing)")
    args = parser.parse_args()

    if len(args.catalogs) != len(args.outputs):
        parser.error(
            f"Got {len(args.catalogs)} catalogs but {len(args.outputs)} outputs — "
            "these must be passed in matching pairs."
        )

    for catalog_path, output_path in zip(args.catalogs, args.outputs):
        print(f"\n=== Processing {catalog_path} ===")
        catalog = pd.read_csv(catalog_path, parse_dates=["time_utc"])
        print(f"Loaded catalog: {catalog.shape}")

        tabular_df = build_tabular_dataset(
            catalog, args.data_dir,
            predict_frame  = args.predict_frame,
            rain_threshold = args.rain_threshold,
            limit          = args.limit,
        )
        print(f"Final tabular shape: {tabular_df.shape}")
        print(tabular_df.describe())
        print(f"target_exceeds distribution:\n{tabular_df['target_exceeds'].value_counts(normalize=True)}")

        tabular_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    #main()

    ### --- EXAMPLE CALL: --- ###
    # python extract_tabular_features.py \
             # --catalogs vil_train_catalog.csv vil_test_catalog.csv \
             # --outputs vil_tabular_features_train.csv vil_tabular_features_test.csv

    results = run_extraction(catalogs=["vil_train_catalog.csv", "vil_test_catalog.csv"],
                             outputs=["vil_tabular_features_train.csv", "vil_tabular_features_test.csv"],
                             limit=None,
    )

    train_df = results["vil_tabular_features_train.csv"]
    train_df.describe()