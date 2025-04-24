import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from wect import ECTConfig, compute_ecc, normalize

# --- WDECTLayer with face-empty guard ---
class WDECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.Tensor):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps)
                  .view(-1,1,1,1),
            requires_grad=False
        )
        self.v = nn.Parameter(v.unsqueeze(0), requires_grad=False)

    def forward(self, batch: Batch):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ self.v.squeeze(0)
        # edges
        eh, _ = nh[batch.edge_index].max(dim=0)
        eh = eh * batch.edge_weights.unsqueeze(1)
        idx_e = batch.batch[batch.edge_index[0]]

        # base ECC (points) minus edges
        ecc_pts = compute_ecc(nh, batch.batch, self.lin)
        ecc_edges = compute_ecc(eh, idx_e, self.lin)
        ecc = ecc_pts - ecc_edges

        # add faces term only if faces exist
        if hasattr(batch, 'face') and batch.face.numel() > 0:
            fh, _ = nh[batch.face].max(dim=0)
            fh = fh * batch.face_weights.unsqueeze(1)
            idx_f = batch.batch[batch.face[0]]
            ecc += compute_ecc(fh, idx_f, self.lin)

        return normalize(ecc) if self.config.normalized else ecc.squeeze()

# --- Classifier remains the same ---
class WDECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim, num_classes, num_directions, dim):
        super().__init__()
        v = torch.randn(dim, num_directions)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        in_dim = num_directions * config.bump_steps
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Batch):
        feat = self.wdect(batch)
        flat = feat.view(batch.num_graphs, -1)
        x = F.relu(self.fc1(flat))
        return self.fc2(x), flat

    def compute_loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats)

# --- Graph creation for a 2-D feature vector ---
def create_complete_graph(f: torch.Tensor) -> Data:
    F_dim = f.size(0)
    x = torch.diag(f)
    # undirected complete graph edges
    edge_index = torch.combinations(torch.arange(F_dim), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # no faces if F_dim < 3
    face = torch.combinations(torch.arange(F_dim), r=3).T if F_dim >= 3 else torch.empty((3,0), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        face=face,
        node_weights=torch.ones(F_dim),
        edge_weights=torch.ones(edge_index.size(1)),
        face_weights=torch.ones(face.size(1)) if face.numel() else torch.empty(0)
    )

# --- Load only moons & circles ---
def load_graphs(name):
    typ, size, noise = name.split("_")
    size = int(size); noise = float(noise)
    X, y = (make_moons if typ=="moons" else make_circles)(
        n_samples=size, noise=noise, random_state=42
    )
    graphs = []
    for xi, yi in zip(X, y):
        g = create_complete_graph(torch.tensor(xi, dtype=torch.float))
        g.y = torch.tensor([yi], dtype=torch.long)
        graphs.append(g)
    return graphs, 2, 2

# --- Training & evaluation functions (unchanged) ---
def train(model, data_list, epochs=100, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        batch = Batch.from_data_list(data_list)
        logits, feats = model(batch)
        loss = model.compute_loss(logits, batch.y, feats)
        loss.backward(); opt.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        logits, _ = model(batch)
        return (logits.argmax(1) == batch.y).float().mean().item()

# --- Main: iterate moons & circles variations ---
if __name__=="__main__":
    sizes = [1000, 5000, 10000, 20000]
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    datasets = [f"{t}_{s}_{n}" for t in ["moons","circles"]
                for s in sizes for n in noises]

    cfg = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    for name in datasets:
        print(f"\n{name}")
        graphs, dim, num_cls = load_graphs(name)
        tr, te = train_test_split(graphs, test_size=0.2, random_state=42)
        model = WDECTClassifier(cfg, hidden_dim=64, num_classes=num_cls, num_directions=64, dim=dim)
        train(model, tr, epochs=50)
        print(" Train Acc:", evaluate(model, tr))
        print(" Test  Acc:", evaluate(model, te))
