import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset import NILMDataset
from model import Seq2Point

CSV = "data/house1_ukdale.csv"
THRESH = {"fridge": 50, "kettle": 1000, "microwave": 200}   # ON hone ki limit (W)
EPOCHS = 8

os.makedirs("models", exist_ok=True)
stats, metrics = {}, {}


def evaluate(model, dl, ds, thr):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for x, y in dl:
            P.append(model(x).numpy())
            T.append(y.numpy())
    pred = np.clip(np.concatenate(P) * ds.y_std + ds.y_mean, 0, None)
    true = np.concatenate(T) * ds.y_std + ds.y_mean
    mae = np.abs(pred - true).mean()
    on_p, on_t = pred > thr, true > thr
    tp = (on_p & on_t).sum()
    fp = (on_p & ~on_t).sum()
    fn = (~on_p & on_t).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    sae = abs(pred.sum() - true.sum()) / max(true.sum(), 1)
    return float(mae), float(f1), float(sae)


for name, thr in THRESH.items():
    print(f"\n===== {name} =====")
    ds = NILMDataset(CSV, name)
    n_train = int(0.8 * len(ds))
    train_dl = DataLoader(Subset(ds, range(0, n_train)), batch_size=128, shuffle=True)
    test_dl = DataLoader(Subset(ds, range(n_train, len(ds))), batch_size=256)

    model = Seq2Point()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        mae, f1, sae = evaluate(model, test_dl, ds, thr)
        print(f"Epoch {epoch + 1}/{EPOCHS} | MAE {mae:.1f} W | F1 {f1:.2f}")

    torch.save(model.state_dict(), f"models/{name}.pt")
    stats[name] = {"x_mean": float(ds.x_mean), "x_std": float(ds.x_std),
                   "y_mean": float(ds.y_mean), "y_std": float(ds.y_std)}
    metrics[name] = {"MAE (W)": round(mae, 1), "F1": round(f1, 2),
                     "Energy error (%)": round(sae * 100, 1)}

json.dump(stats, open("models/stats.json", "w"), indent=2)
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print("\nSab models save ho gaye.")