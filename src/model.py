import torch
import torch.nn as nn

class Seq2Point(nn.Module):
    def __init__(self, window=99):
        super().__init__()
        # Conv layers: power ke pattern (jaise on/off ka shape) pehchanti hain
        self.conv = nn.Sequential(
            nn.Conv1d(1, 30, 10), nn.ReLU(),
            nn.Conv1d(30, 30, 8), nn.ReLU(),
            nn.Conv1d(30, 40, 6), nn.ReLU(),
            nn.Conv1d(40, 50, 5), nn.ReLU(),
        )
        out_len = window - (10 + 8 + 6 + 5) + 4
        # Linear layers: pattern se final power number nikalti hain
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * out_len, 1024), nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)


if __name__ == "__main__":
    model = Seq2Point()
    dummy = torch.randn(8, 1, 99)   # 8 nakli samples
    out = model(dummy)
    print("Output shape:", out.shape)