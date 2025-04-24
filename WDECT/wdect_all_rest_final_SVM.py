import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from wect import ECTConfig, compute_ecc, normalize

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 1) Graph creation for 2D features
# -----------------------------------------------------------------------------
def create_complete_graph(f: torch.Tensor) -> Data:
    F_dim = f.size(0)
    x = torch.diag(f).to(device)
    # complete undirected edges
    ei = torch.combinations(torch.arange(F_dim, device=device), r=2).T
    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    # triangles (faces) if F_dim >=3
    face = (torch.combinations(torch.arange(F_dim, device=device), r=3).T
            if F_dim >= 3 else torch.empty((3,0), dtype=torch.long, device=device))
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
# 2) WDECTLayer with face-empty guard
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
        # Points ECC
        ecc_pts = compute_ecc(nh, batch.batch, self.lin)
        # Edges term
        eh, _ = nh[batch.edge_index].max(dim=0)
        eh = eh * batch.edge_weights.unsqueeze(1)
        idx_e = batch.batch[batch.edge_index[0]]
        ecc_edge = compute_ecc(eh, idx_e, self.lin)
        ecc = ecc_pts - ecc_edge
        # Faces term (if present)
        if hasattr(batch, 'face') and batch.face.numel() > 0:
            fh, _ = nh[batch.face].max(dim=0)
            fh = fh * batch.face_weights.unsqueeze(1)
            idx_f = batch.batch[batch.face[0]]
            ecc += compute_ecc(fh, idx_f, self.lin)
        return normalize(ecc) if self.config.normalized else ecc.squeeze()

# -----------------------------------------------------------------------------
# 3) RFF mapping to approximate RBF kernel
# -----------------------------------------------------------------------------
class RFFMapping(nn.Module):
    def __init__(self, in_dim, rff_dim, gamma):
        super().__init__()
        W = torch.randn(in_dim, rff_dim, device=device) * math.sqrt(2 * gamma)
        b = 2 * math.pi * torch.rand(rff_dim, device=device)
        self.register_buffer('W', W)
        self.register_buffer('b', b)

    def forward(self, x):
        proj = x @ self.W  # [N, rff_dim]
        return math.sqrt(2.0 / self.W.size(1)) * torch.cos(proj + self.b)

# -----------------------------------------------------------------------------
# 4) Combined “SVM” module
# -----------------------------------------------------------------------------
class WDECT_RBFSVM(nn.Module):
    def __init__(self, config: ECTConfig, dim, num_directions,
                 bump_steps, rff_dim, gamma, num_classes):
        super().__init__()
        # random unit directions for WDECT
        v = torch.randn(dim, num_directions, device=device)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        feat_dim = num_directions * bump_steps
        self.rff    = RFFMapping(feat_dim, rff_dim, gamma)
        self.linear = nn.Linear(rff_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)                      # [B, D, S]
        flat  = wfeat.view(batch.num_graphs, -1)       # [B, feat_dim]
        phi   = self.rff(flat)                         # [B, rff_dim]
        return self.linear(phi), flat                  # logits, raw WDECT feats

    def loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# -----------------------------------------------------------------------------
# 5) Data loader: moons & circles variations
# -----------------------------------------------------------------------------
def load_graphs(name: str):
    typ, size, noise = name.split("_")
    size, noise = int(size), float(noise)
    X, y = (make_moons if typ=="moons" else make_circles)(
        n_samples=size, noise=noise, random_state=42
    )
    graphs = []
    for xi, yi in zip(X, y):
        g = create_complete_graph(torch.tensor(xi, dtype=torch.float, device=device))
        g.y = torch.tensor([yi], dtype=torch.long, device=device)
        graphs.append(g)
    return graphs

# -----------------------------------------------------------------------------
# 6) Training & evaluation
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
# 7) Main script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sizes  = [1000, 5000, 10000, 20000]
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    datasets = [f"{t}_{s}_{n}" for t in ["moons","circles"]
                for s in sizes for n in noises]

    # WDECT config
    config = ECTConfig(
        radius=1.0, bump_steps=16,
        ect_type="faces", normalized=True, fixed=True
    )
    # SVM/RFF hyperparams
    num_dirs, rff_dim, gamma = 64, 512, 1.0
    hidden = None  # not used

    for name in datasets:
        print(f"\nDataset: {name}")
        graphs = load_graphs(name)
        tr, te = train_test_split(graphs, test_size=0.2, random_state=42)
        model = WDECT_RBFSVM(
            config=config,
            dim=2,
            num_directions=num_dirs,
            bump_steps=config.bump_steps,
            rff_dim=rff_dim,
            gamma=gamma,
            num_classes=2
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train
        for epoch in range(1, 51):
            loss = train_epoch(model, tr, optimizer)
            if epoch % 10 == 0:
                print(f" Epoch {epoch:02d}, Loss: {loss:.4f}")

        # Evaluate
        tr_acc = evaluate(model, tr)
        te_acc = evaluate(model, te)
        print(f" Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
