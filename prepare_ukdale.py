import tables
import numpy as np
import pandas as pd

H5 = "data/ukdale/ukdale.h5"
METERS = {"aggregate": 1, "kettle": 10, "fridge": 12, "microwave": 13}
DAYS = 60

series = {}
with tables.open_file(H5, mode="r") as f:
    for name, m in METERS.items():
        node = f.get_node(f"/building1/elec/meter{m}/table")
        t = node.read()
        if m == 1:
            print("Table ke fields:", t.dtype.names)
        idx = pd.to_datetime(t["index"], unit="ns")
        vals = t["values_block_0"]
        print(f"{name:10s} meter{m:<3d} rows={len(t):>9,}  columns={vals.shape[1]}  "
              f"{idx.min()}  ->  {idx.max()}")
        s = pd.Series(vals[:, 0].astype("float32"), index=idx)
        series[name] = s[~s.index.duplicated()]

start = max(s.index.min() for s in series.values())
end = start + pd.Timedelta(days=DAYS)
print("\nUse hone wala period:", start, "se", end)

out = {}
for name, s in series.items():
    out[name] = s[(s.index >= start) & (s.index < end)].resample("1min").mean()

df = pd.DataFrame(out).ffill(limit=5).dropna()
df.index.name = "timestamp"
df = df.reset_index()
df.to_csv("data/house1_ukdale.csv", index=False)

print("\nBan gaya! Rows:", len(df), "| Din:", round(len(df) / 1440, 1))
print(df.describe().round(1))