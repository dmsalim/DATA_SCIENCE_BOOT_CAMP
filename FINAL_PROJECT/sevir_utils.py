import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import h5py
import torch 
from   torch.utils.data  import Dataset
import pdb

### PYTORCH DATASET WRAPPER ###

class SEVIRVILDataset(Dataset):
    def __init__(self, catalog, data_dir=os.path.join(os.getcwd(), "data"), input_len=13, output_len=12, stride=12):
        self.data_dir = data_dir
        self.input_len = input_len
        self.output_len = output_len
        self.index = []  # list of (file_name, file_index, window_start)

        for _, row in catalog.iterrows():
            file_name = row["file_name"]
            file_index = row["file_index"]
            T = 49  # confirmed earlier — all SEVIR VIL events have 49 time steps
            for start in range(0, T - input_len - output_len + 1, stride):
                self.index.append((file_name, file_index, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_name, file_index, start = self.index[idx]
        seq = load_event(self.data_dir, file_name, file_index)  # (384, 384, 49)

        x = seq[:, :, start : start + self.input_len]
        y = seq[:, :, start + self.input_len : start + self.input_len + self.output_len]

        x = torch.from_numpy(x).float() / 255.0 * 2 - 1
        y = torch.from_numpy(y).float() / 255.0 * 2 - 1

        return x.permute(2, 0, 1), y.permute(2, 0, 1)  # (T, H, W)

### BUILD WINDOWING FUNCITON (13-INPUT, 12-OUTPUT, STRIDE=12) ###
def make_windows(vil_sequence, input_len=13, output_len=12, stride=12):
    # --- vil_sequence: array of shape (H, W, T) --- #
    T = vil_sequence.shape[-1]
    windows = []
    for start in range(0, T - input_len - output_len + 1, stride):
        x = vil_sequence[:, :, start : start + input_len]
        y = vil_sequence[:, :, start + input_len : start + input_len + output_len]
        windows.append((x, y))
    return windows

def load_event(data_dir, file_name, file_index):
    """Load a single VIL event tensor from a SEVIR HDF5 file.

    Returns array of shape (384, 384, 49).
    """
    file_path = os.path.join(data_dir, file_name)
    with h5py.File(file_path, "r") as f:
        seq = f["vil"][int(file_index)]
    return seq