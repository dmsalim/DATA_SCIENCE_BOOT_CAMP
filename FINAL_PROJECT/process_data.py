import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import h5py
import torch 
from   torch.utils.data  import Dataset
from   sevir_utils       import SEVIRVILDataset, make_windows
import pdb



catalog = pd.read_csv("CATALOG.csv", parse_dates=["time_utc"], low_memory=False)
print(catalog.shape)
print(catalog["img_type"].value_counts())
print(catalog[catalog["img_type"] == "vil"][["id", "file_name", "file_index", "time_utc"]].head())

data_path    = os.path.join(os.getcwd(), "data")
vil_row      = catalog[catalog["img_type"] == "vil"].iloc[0]
vil_row_file = os.path.join(data_path, vil_row["file_name"])

with h5py.File(vil_row_file, "r") as f:
    idx = int(vil_row["file_index"])
    vil_data = f["vil"][idx]          # shape (384, 384, 49)
    print(vil_data.shape)
    print(vil_data.dtype, vil_data.min(), vil_data.max())

### VISUALISE A SEQUENCE OF VIL FRAMES ###
step     = 8
n_frames = vil_data.shape[-1]          # 49
n_plots  = int(np.ceil(n_frames / step))  # 7

fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
for ax, t in zip(axes, range(0, n_frames, step)):
    ax.imshow(vil_data[:, :, t], cmap="magma", vmin=0, vmax=255)
    ax.set_title(f"t={t}")
    ax.axis("off")
fig.savefig("TESTS/vil_sample.pdf", bbox_inches="tight")

for t in range(0, n_frames):
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(vil_data[:, :, t], cmap="magma", vmin=0, vmax=255)
    ax.set_title(f"t={t}")
    ax.axis("off")
    f.savefig(f"TESTS/MOVIE/vil_sample_movie_{t:04d}.pdf", bbox_inches="tight")

### TRAIN/TEST SPLIT BY DATE ###
train_catalog = catalog[(catalog["img_type"] == "vil") & (catalog["time_utc"] < "2019-06-01")]
test_catalog = catalog[(catalog["img_type"] == "vil") & (catalog["time_utc"] >= "2019-06-01")]
print(f"Train events: {len(train_catalog)}, Test events: {len(test_catalog)}")

train_catalog.to_csv("vil_train_catalog.csv", index=False)
test_catalog.to_csv("vil_test_catalog.csv", index=False)


