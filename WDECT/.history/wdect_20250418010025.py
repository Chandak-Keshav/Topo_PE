import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ect import ECTConfig, compute_ecc, normalize
from build_weighted_complex import build_weighted_complex

# -----------------------------------------------------------------------------
# 1) WDECTLayer: weighted + differentiable
# -----------------------------------------------------------------------------
class WDECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.Tensor):
        super().__init__()
        self.config = config
        # thresholds lin: [bump_steps,1,1,1]
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps)
                 .view(-1,1,1,1),
            requires_grad=False
        )
        # fixed directions v: [2, num_directions] -> [1,2,D]
        self.v = nn.Parameter(v.unsqueeze(0), requires_grad=False)

        # choose point/edge/face compute with weighting
        if config.ect_type == "points":
            self._compute = self._compute_points
        elif config.ect_type == "edges":
            self._compute = self._compute_edges
        else:
            self._compute = self._compute_faces

    def forward(self, batch: Batch):
        # compute WDECT: [B, D, S]
        ect = self._compute(batch, self.v, self.lin)
        return normalize(ect) if self.config.normalized else ect.squeeze()

    def _compute_points(self, batch, v, lin):
        # node heights weighted by node_weights
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        return compute_ecc(nh, batch.batch, lin)

    def _compute_edges(self, batch, v, lin):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        eh, _ = nh[batch.edge_index].max(dim=0)
        # apply edge_weights per edge
        eh = eh * batch.edge_weights
        return compute_ecc(nh, batch.batch, lin) - compute_ecc(eh, batch.batch[batch.edge_index[0]], lin)

    def _compute_faces(self, batch, v, lin):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        eh, _ = nh[batch.edge_index].max(dim=0)
        fh, _ = nh[batch.face].max(dim=0)
        eh = eh * batch.edge_weights
        fh = fh * batch.face_weights
        idx_e = batch.batch[batch.edge_index[0]]
        idx_f = batch.batch[batch.face[0]]
        return (
            compute_ecc(nh, batch.batch, lin)
            - compute_ecc(eh, idx_e, lin)
            + compute_ecc(fh, idx_f, lin)
        )

# -----------------------------------------------------------------------------
# 2) Classifier wrapping WDECT
# -----------------------------------------------------------------------------
class WDECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int, num_directions: int):
        super().__init__()
        self.config = config
        # generate unit‐circle directions [2, D]
        angles = torch.arange(num_directions, dtype=torch.float) * (2*math.pi/num_directions)
        v = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)

        self.wdect = WDECTLayer(config, v)
        in_dim = num_directions * config.bump_steps
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)             # [B, D, S]
        flat  = wfeat.view(batch.num_graphs, -1)
        x     = F.relu(self.fc1(flat))
        return self.fc2(x), flat

    def compute_loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# -----------------------------------------------------------------------------
# 3) Data loader for load_digits
# -----------------------------------------------------------------------------
def load_digits_data():
    data = load_digits()
    imgs, labels = data.images, data.target
    list_data = []
    for img, lbl in zip(imgs, labels):
        V,E,F,Vw,Ew,Fw = build_weighted_complex(img)
        data_obj = Data(
            x=torch.tensor(V, dtype=torch.float),
            edge_index=torch.tensor(E.T, dtype=torch.long) if E.size else torch.empty((2,0),dtype=torch.long),
            face=torch.tensor(F.T, dtype=torch.long)       if F.size else torch.empty((3,0),dtype=torch.long),
            node_weights=torch.tensor(Vw, dtype=torch.float),
            edge_weights=torch.tensor(Ew, dtype=torch.float),
            face_weights=torch.tensor(Fw, dtype=torch.float),
            y=torch.tensor([lbl], dtype=torch.long)
        )
        list_data.append(data_obj)
    return list_data

# -----------------------------------------------------------------------------
# 4) Train & Evaluate
# -----------------------------------------------------------------------------
def train(model, data_list, epochs=100, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        batch = Batch.from_data_list(data_list)
        logits, feats = model(batch)
        loss = model.compute_loss(logits, batch.y, feats)
        loss.backward()
        opt.step()
        if epoch%10==0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        return (pred == batch.y).float().mean().item()

# -----------------------------------------------------------------------------
# 5) Main
# -----------------------------------------------------------------------------
if __name__=="__main__":
    config = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    num_dirs, hid_dim, n_cls = 64, 64, 10

    full_data = load_digits_data()
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

    model = WDECTClassifier(config, hid_dim, n_cls, num_dirs)
    train(model, train_data, epochs=200)
    print("Train Acc:", evaluate(model, train_data))
    print("Test  Acc:", evaluate(model, test_data))
