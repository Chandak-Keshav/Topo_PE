import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from wect import ECTConfig, compute_ecc, normalize
from build_weighted_complex import build_weighted_complex
import pandas as pd

# -----------------------------------------------------------------------------
# 0) Device
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 1) Graph creation for vector features
# -----------------------------------------------------------------------------
def create_complete_graph(f: torch.Tensor) -> Data:
    F_dim = f.size(0)
    x = torch.diag(f).to(device)
    ei = torch.combinations(torch.arange(F_dim, device=device), r=2).T
    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    face = torch.combinations(torch.arange(F_dim, device=device), r=3).T \
           if F_dim >= 3 else torch.empty((3,0), dtype=torch.long, device=device)
    return Data(
        x=x,
        edge_index=edge_index,
        face=face,
        node_weights=torch.ones(F_dim, device=device),
        edge_weights=torch.ones(edge_index.size(1), device=device),
        face_weights=(torch.ones(face.size(1), device=device)
                      if face.numel() else torch.empty(0, device=device))
    )

# -----------------------------------------------------------------------------
# 2) WDECTLayer (with empty-face guard)
# -----------------------------------------------------------------------------
class WDECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.Tensor):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps)
                 .view(-1,1,1,1), requires_grad=False
        )
        self.v = nn.Parameter(v.unsqueeze(0), requires_grad=False)

    def forward(self, batch: Batch):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ self.v.squeeze(0)
        ecc_pts = compute_ecc(nh, batch.batch, self.lin)
        eh, _ = nh[batch.edge_index].max(dim=0)
        eh = eh * batch.edge_weights.unsqueeze(1)
        idx_e = batch.batch[batch.edge_index[0]]
        ecc_edge = compute_ecc(eh, idx_e, self.lin)
        ecc = ecc_pts - ecc_edge
        if hasattr(batch, 'face') and batch.face.numel() > 0:
            fh, _ = nh[batch.face].max(dim=0)
            fh = fh * batch.face_weights.unsqueeze(1)
            idx_f = batch.batch[batch.face[0]]
            ecc += compute_ecc(fh, idx_f, self.lin)
        return normalize(ecc) if self.config.normalized else ecc.squeeze()

# -----------------------------------------------------------------------------
# 3) RFF for RBF-kernel approximation
# -----------------------------------------------------------------------------
class RFFMapping(nn.Module):
    def __init__(self, in_dim, rff_dim, gamma):
        super().__init__()
        W = torch.randn(in_dim, rff_dim, device=device) * math.sqrt(2 * gamma)
        b = 2 * math.pi * torch.rand(rff_dim, device=device)
        self.register_buffer('W', W)
        self.register_buffer('b', b)

    def forward(self, x):
        proj = x @ self.W
        return math.sqrt(2.0 / self.W.size(1)) * torch.cos(proj + self.b)

# -----------------------------------------------------------------------------
# 4) SVM-like classifier module
# -----------------------------------------------------------------------------
class WDECT_RBFSVM(nn.Module):
    def __init__(self, config: ECTConfig, dim, num_directions,
                 rff_dim, gamma, num_classes):
        super().__init__()
        v = torch.randn(dim, num_directions, device=device)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        feat_dim = num_directions * config.bump_steps
        self.rff    = RFFMapping(feat_dim, rff_dim, gamma)
        self.linear = nn.Linear(rff_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)                    # [B, D, S]
        flat  = wfeat.view(batch.num_graphs, -1)     # [B, feat_dim]
        phi   = self.rff(flat)                       # [B, rff_dim]
        return self.linear(phi), flat

    def loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# -----------------------------------------------------------------------------
# 5) Data loader for all datasets
# -----------------------------------------------------------------------------
def load_and_prepare_dataset(name: str):
    if name == "load_digits":
        data = load_digits()
        data_list = []
        for img, lbl in zip(data.images, data.target):
            V, E, F, Vw, Ew, Fw = build_weighted_complex(img)
            data_list.append(Data(
                x=torch.tensor(V, dtype=torch.float, device=device),
                edge_index=(torch.tensor(E.T, dtype=torch.long, device=device)
                            if E.size else torch.empty((2,0),dtype=torch.long,device=device)),
                face=(torch.tensor(F.T, dtype=torch.long, device=device)
                      if F.size else torch.empty((3,0),dtype=torch.long,device=device)),
                node_weights=torch.tensor(Vw, dtype=torch.float, device=device),
                edge_weights=torch.tensor(Ew, dtype=torch.float, device=device),
                face_weights=torch.tensor(Fw, dtype=torch.float, device=device),
                y=torch.tensor([lbl], dtype=torch.long, device=device)
            ))
        dim, num_classes = 2, 10

    elif name == "iris":
        data = load_iris()
        feats = torch.tensor(data.data, dtype=torch.float, device=device)
        labs = torch.tensor(data.target, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes = feats.shape[1], 3

    elif name == "spect":
        data = fetch_openml(name='SPECT', version=1, parser='auto')
        data.data = data.data.apply(lambda x: x.cat.codes if pd.api.types.is_categorical_dtype(x) else x)
        feats = torch.tensor(data.data.to_numpy(), dtype=torch.float, device=device)
        labs  = torch.tensor((data.target=='1').astype(int), dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes = feats.shape[1], 2

    elif name == "letters":
        data = fetch_openml(name='letter', version=1, parser='auto')
        feats = torch.tensor(data.data.to_numpy(), dtype=torch.float, device=device)
        labs  = torch.tensor(data.target.map(lambda x: ord(x)-65), dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes = feats.shape[1], 26

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return data_list, dim, num_classes

# -----------------------------------------------------------------------------
# 6) Training & evaluation helpers
# -----------------------------------------------------------------------------
def train_epoch(model, data_list, optimizer):
    model.train(); optimizer.zero_grad()
    batch = Batch.from_data_list(data_list).to(device)
    logits, feats = model(batch)
    loss = model.loss(logits, batch.y, feats)
    loss.backward(); optimizer.step()
    return loss.item()

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        logits, _ = model(batch)
        return (logits.argmax(dim=1) == batch.y).float().mean().item()

# -----------------------------------------------------------------------------
# 7) Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = ["iris", "spect", "load_digits", "letters"]
    config = ECTConfig(radius=1.0, bump_steps=16,
                       ect_type="faces", normalized=True, fixed=True)
    # SVM/RBF hyperparameters
    num_dirs, rff_dim, gamma = 64, 512, 1.0
    lr, epochs = 1e-2, 100

    for name in datasets:
        print(f"\n--- {name} ---")
        data_list, dim, num_cls = load_and_prepare_dataset(name)
        tr, te = train_test_split(data_list, test_size=0.2, random_state=42)
        model = WDECT_RBFSVM(config, dim, num_dirs, rff_dim, gamma, num_cls).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for ep in range(1, epochs+1):
            loss = train_epoch(model, tr, opt)
            if ep % 10 == 0:
                print(f"Epoch {ep:02d}, Loss: {loss:.4f}")

        tr_acc = evaluate(model, tr)
        te_acc = evaluate(model, te)
        print(f"Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
