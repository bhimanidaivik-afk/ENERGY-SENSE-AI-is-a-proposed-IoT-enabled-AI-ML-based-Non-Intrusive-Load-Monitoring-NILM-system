import numpy as np
import pandas as pd

np.random.seed(42)
days = 3
n = days * 24 * 60   # 1 minute ka data, 3 din

# Fridge: baar-baar on/off hota hai
fridge = np.zeros(n)
t = 0
while t < n:
    t += np.random.randint(15, 30)       # off time
    on = np.random.randint(8, 15)        # on time
    fridge[t:t + on] = np.random.uniform(120, 160)
    t += on

# Kettle: din mein 4 baar, 3 minute, 2000 W
kettle = np.zeros(n)
for d in range(days):
    for _ in range(4):
        s = d * 1440 + np.random.randint(360, 1320)
        kettle[s:s + 3] = 2000

# Microwave: din mein 2 baar, 5 minute, 1000 W
microwave = np.zeros(n)
for d in range(days):
    for _ in range(2):
        s = d * 1440 + np.random.randint(360, 1320)
        microwave[s:s + 5] = 1000

# Baaki ghar ka chhota load (bulb, fan etc.)
other = np.random.uniform(40, 80, n)

aggregate = fridge + kettle + microwave + other

df = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
    "aggregate": aggregate,
    "fridge": fridge,
    "kettle": kettle,
    "microwave": microwave,
})
df.to_csv("data/house1.csv", index=False)
print("Data ban gaya! Rows:", len(df))