import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class NILMDataset(Dataset):
    def __init__(self, csv_path, appliance, window=99):
        df = pd.read_csv(csv_path)
        self.x = df["aggregate"].values.astype(np.float32)
        self.y = df[appliance].values.astype(np.float32)

        # normalize (numbers ko chhota karna, taaki model aasani se seekhe)
        self.x_mean, self.x_std = self.x.mean(), self.x.std()
        self.y_mean, self.y_std = self.y.mean(), self.y.std()
        self.x = (self.x - self.x_mean) / self.x_std
        self.y = (self.y - self.y_mean) / self.y_std

        self.window = window
        self.half = window // 2

    def __len__(self):
        return len(self.x) - self.window

    def __getitem__(self, i):
        seq = self.x[i : i + self.window]     # 99 minute ka total power
        target = self.y[i + self.half]        # beech wale minute ka appliance power
        return torch.tensor(seq).unsqueeze(0), torch.tensor(target)


if __name__ == "__main__":
    ds = NILMDataset("data/house1.csv", "fridge")
    x, y = ds[0]
    print("Total samples:", len(ds))
    print("Input shape:", x.shape)
    print("Target:", y)