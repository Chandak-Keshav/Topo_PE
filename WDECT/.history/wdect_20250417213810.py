import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT, weighted_euler_curve
from distance_RotationInvariant import distance_RotationInvariant
from ect import ECTConfig, normalize
from wect import WECTLayer

# -----------------------------------------------------------------------------
# 1) Data loading: same as before
# -----------------------------------------------------------------------------
def load_digits_weighted_complex_data():
    data = load_digits()
    images, labels = data.images, data.target
    data_list = []
    for img, lbl in zip(images, labels):
        V, E, F, Vw, Ew, Fw = build_weighted_complex(img)
        x = torch.tensor(V, dtype=torch.float)
        node_weights = torch.tensor(Vw, dtype=torch.float)
        edge_index = torch.tensor(E.T, dtype=torch.long) if E.size else torch.empty((2,0),dtype=torch.long)
        face       = torch.tensor(F.T, dtype=torch.long) if F.size else torch.empty((3,0),dtype=torch.long)
        data_list.append(Data(
            x=x,
            node_weights=node_weights,
            edge_index=edge_index,
            face=face,
            y=torch.tensor([lbl],dtype=torch.long)
        ))
    return data_list

# -----------------------------------------------------------------------------
# 2) Classifier wrapping WECT → WDECT
# -----------------------------------------------------------------------------
class WECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int, num_directions: int):
        super().__init__()
        self.config = config

        # ── Generate directions v of shape [2, num_directions] on the unit circle ──
        angles = torch.arange(num_directions, dtype=torch.float) * (2 * math.pi / num_directions)
        v = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)  # [2, num_directions]

        # ── WECTLayer expects v like this ──
        self.wect_layer = WECTLayer(config, v)

        # ── Build classifier MLP ──
        in_dim = num_directions * config.bump_steps
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Batch):
        wect_out = self.wect_layer(batch)               # [B, D, S]
        feats    = wect_out.view(batch.num_graphs, -1)   # [B, D·S]
        x        = F.relu(self.fc1(feats))
        return self.fc2(x), feats

    def compute_loss(self, logits, targets, feats, λ=0.1):
        ce  = F.cross_entropy(logits, targets)
        reg = torch.norm(feats, p=2)
        return ce + λ * reg

# -----------------------------------------------------------------------------
# 3) Training & eval (unchanged)
# -----------------------------------------------------------------------------
def train(model, train_data, num_epochs=200, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        batch = Batch.from_data_list(train_data)
        logits, feats = model(batch)
        loss = model.compute_loss(logits, batch.y, feats)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        acc = (pred == batch.y).float().mean().item()
    return acc

# -----------------------------------------------------------------------------
# 4) Main: wire it all up with correct args
# -----------------------------------------------------------------------------
def main():
    # config no longer takes num_directions
    config = ECTConfig(
        radius=1.0,
        bump_steps=16,
        ect_type="faces",
        normalized=True,
        fixed=True
    )

    num_directions = 128
    hidden_dim = 64
    num_classes = 10
    num_epochs = 200

    data_list = load_digits_weighted_complex_data()
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

    model = WECTClassifier(config, hidden_dim, num_classes, num_directions)
    train(model, train_data, num_epochs=num_epochs)

    print("Train  Acc:", evaluate(model, train_data))
    print("Test   Acc:", evaluate(model, test_data))

    # visualize one sample WDECT
    batch = Batch.from_data_list(test_data)
    wout, _ = model.wect_layer(batch)
    sample = wout[0].cpu().detach().numpy().reshape(num_directions, config.bump_steps)
    plt.imshow(sample, aspect='auto', cmap='viridis')
    plt.xlabel("Threshold Steps")
    plt.ylabel("Directions")
    plt.title("Sample WDECT Representation")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
