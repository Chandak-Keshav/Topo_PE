import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wect import ECTConfig, compute_ecc, normalize

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Graph creation for 2D features
def create_complete_graph(f: torch.Tensor) -> Data:
    F_dim = f.size(0)
    x = torch.diag(f).to(device)
    ei = torch.combinations(torch.arange(F_dim, device=device), r=2).T
    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
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

# WDECTLayer with face-empty guard
class WDECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.Tensor):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps, device=device)
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

# RFF mapping to approximate RBF kernel
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

# Combined “SVM” module
class WDECT_RBFSVM(nn.Module):
    def __init__(self, config: ECTConfig, dim, num_directions,
                 bump_steps, rff_dim, gamma, num_classes):
        super().__init__()
        v = torch.randn(dim, num_directions, device=device)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        feat_dim = num_directions * bump_steps
        self.rff = RFFMapping(feat_dim, rff_dim, gamma)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(rff_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)
        flat = wfeat.view(batch.num_graphs, -1)
        phi = self.rff(flat)
        phi = self.dropout(phi)
        return self.linear(phi), flat

    def loss(self, logits, targets, feats, λ=0.5):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# Data loader: moons & circles variations
def load_graphs(name: str, scaler: StandardScaler):
    typ, size, noise = name.split("_")
    size, noise = int(size), float(noise)
    X, y = (make_moons if typ=="moons" else make_circles)(
        n_samples=size, noise=noise, random_state=42
    )
    X = scaler.fit_transform(X)
    graphs = []
    for xi, yi in zip(X, y):
        g = create_complete_graph(torch.tensor(xi, dtype=torch.float, device=device))
        g.y = torch.tensor([yi], dtype=torch.long, device=device)
        graphs.append(g)
    return graphs

# Training & evaluation
def train_epoch(model, data_list, optimizer, scheduler=None):
    model.train()
    optimizer.zero_grad()
    batch = Batch.from_data_list(data_list).to(device)
    logits, feats = model(batch)
    loss = model.loss(logits, batch.y, feats)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step(loss)
    return loss.item()

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        logits, _ = model(batch)
        return (logits.argmax(dim=1) == batch.y).float().mean().item()

def train_with_early_stopping(model, train_list, val_list, optimizer, scheduler, epochs, patience=20):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs+1):
        loss = train_epoch(model, train_list, optimizer, scheduler)
        val_batch = Batch.from_data_list(val_list).to(device)
        with torch.no_grad():
            val_logits, val_feats = model(val_batch)
            val_loss = model.loss(val_logits, val_batch.y, val_feats).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

# Main script
if __name__ == "__main__":
    sizes = [1000, 5000, 10000, 20000]
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    datasets = [f"{t}_{s}_{n}" for t in ["moons", "circles"]
                for s in sizes for n in noises]
    config = ECTConfig(
        radius=1.0, bump_steps=16,
        ect_type="faces", normalized=True, fixed=True
    )
    num_dirs, rff_dim, gamma = 32, 256, 0.5
    lr, epochs = 1e-2, 200
    scaler = StandardScaler()

    accuracies = []
    for name in datasets:
        print(f"\nDataset: {name}")
        graphs = load_graphs(name, scaler)
        train_val, test = train_test_split(graphs, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        model = WDECT_RBFSVM(
            config=config,
            dim=2,
            num_directions=num_dirs,
            bump_steps=config.bump_steps,
            rff_dim=rff_dim,
            gamma=gamma,
            num_classes=2
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        train_with_early_stopping(model, train, val, optimizer, scheduler, epochs)

        tr_acc = evaluate(model, train)
        te_acc = evaluate(model, test)
        print(f"Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
        accuracies.append(f"Dataset: {name}, Train Acc: {tr_acc:.4f}, Test Acc: {te_acc:.4f}")

    # Save accuracies to file
    with open("moons_circles_accuracies.txt", "w") as f:
        for acc in accuracies:
            f.write(acc + "\n")