import torch
import matplotlib.pyplot as plt
from dataset import NILMDataset
from model import Seq2Point

APPLIANCE = "fridge"   # train.py mein jo rakha tha wahi

ds = NILMDataset("data/house1.csv", APPLIANCE)

# Trained model load karo
model = Seq2Point()
model.load_state_dict(torch.load(f"{APPLIANCE}_model.pt"))
model.eval()

# Test wale hisse ke 300 minute ka prediction nikalo
n_train = int(0.8 * len(ds))
start, count = n_train, 300

actual, predicted, total = [], [], []
with torch.no_grad():
    for i in range(start, start + count):
        x, y = ds[i]
        pred = model(x.unsqueeze(0)).item()
        # normalize hata kar wapas watt mein badlo
        predicted.append(pred * ds.y_std + ds.y_mean)
        actual.append(y.item() * ds.y_std + ds.y_mean)
        total.append(x[0, ds.half].item() * ds.x_std + ds.x_mean)

# Graph
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(total, color="gray")
ax1.set_title("Ghar ka total power (jo meter dikhata hai)")
ax1.set_ylabel("Watt")

ax2.plot(actual, label="Asli (Actual)", color="green")
ax2.plot(predicted, label="Model ka andaza (Predicted)", color="red", linestyle="--")
ax2.set_title(f"{APPLIANCE.capitalize()} ka power")
ax2.set_xlabel("Minute")
ax2.set_ylabel("Watt")
ax2.legend()

plt.tight_layout()
plt.savefig("result.png", dpi=150)
print("Graph save ho gaya: result.png")
plt.show()
