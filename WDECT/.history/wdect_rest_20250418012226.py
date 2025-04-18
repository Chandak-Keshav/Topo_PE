import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from ect import ECTConfig, compute_ecc, normalize
from build_weighted_complex import build_weighted_complex

# -----------------------------------------------------------------------------
# 1) Generic tabular → weighted complex loader (k‑NN distance grid)
# -----------------------------------------------------------------------------
def compute_distance_grid(X, indices, i):
    nbrs = X[indices[i]]
    D = cdist(nbrs, nbrs)
    mn, mx = D.min(), D.max()
    return (D - mn) / (mx - mn) if mx > mn else D*0.0

def load_tabular_as_complex(X, y, k=None):
    n, f = X.shape
    k = k or min(50, max(5, n//10))
    nn_model = NearestNeighbors(n_neighbors=k).fit(X)
    _, idx = nn_model.kneighbors(X)
    data_list = []
    for i in range(n):
        grid = compute_distance_grid(X, idx, i)
        V,E,F,Vw,Ew,Fw = build_weighted_complex(grid)
        data_list.append(Data(
            x=torch.tensor(V, dtype=torch.float),
            edge_index=torch.tensor(E.T, dtype=torch.long) if E.size else torch.empty((2,0),dtype=torch.long),
            face=torch.tensor(F.T, dtype=torch.long) if F.size else torch.empty((3,0),dtype=torch.long),
            node_weights=torch.tensor(Vw, dtype=torch.float),
            edge_weights=torch.tensor(Ew, dtype=torch.float),
            face_weights=torch.tensor(Fw, dtype=torch.float),
            y=torch.tensor([y[i]], dtype=torch.long)
        ))
    return data_list

def load_digits_data():    
    D = load_digits()
    return load_tabular_as_complex(D.images.reshape(len(D.images), -1), D.target)

def load_iris_data():
    D = load_iris()
    return load_tabular_as_complex(D.data, D.target)

def load_breast_cancer_data():
    D = load_breast_cancer()
    return load_tabular_as_complex(D.data, D.target)

def load_letters_data():
    D = fetch_openml('letter', version=1)
    X = D.data.to_numpy()
    y = LabelEncoder().fit_transform(D.target)
    return load_tabular_as_complex(X, y)

# -----------------------------------------------------------------------------
# 2) WDECTLayer & Classifier (as before)
# -----------------------------------------------------------------------------
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
        if config.ect_type == "points":
            self._compute = self._compute_points
        elif config.ect_type == "edges":
            self._compute = self._compute_edges
        else:
            self._compute = self._compute_faces

    def forward(self, batch: Batch):
        ect = self._compute(batch, self.v, self.lin)
        return normalize(ect) if self.config.normalized else ect.squeeze()

    def _compute_points(self, batch, v, lin):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        return compute_ecc(nh, batch.batch, lin)

    def _compute_edges(self, batch, v, lin):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        eh, _ = nh[batch.edge_index].max(dim=0)
        eh = eh * batch.edge_weights.unsqueeze(1)
        idx_e = batch.batch[batch.edge_index[0]]
        return compute_ecc(nh, batch.batch, lin) - compute_ecc(eh, idx_e, lin)

    def _compute_faces(self, batch, v, lin):
        nh = (batch.x * batch.node_weights.unsqueeze(1)) @ v.squeeze(0)
        eh, _ = nh[batch.edge_index].max(dim=0)
        fh, _ = nh[batch.face].max(dim=0)
        eh = eh * batch.edge_weights.unsqueeze(1)
        fh = fh * batch.face_weights.unsqueeze(1)
        idx_e = batch.batch[batch.edge_index[0]]
        idx_f = batch.batch[batch.face[0]]
        return (
            compute_ecc(nh, batch.batch, lin)
            - compute_ecc(eh, idx_e, lin)
            + compute_ecc(fh, idx_f, lin)
        )

class WDECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int, num_directions: int):
        super().__init__()
        angles = torch.arange(num_directions, dtype=torch.float) * (2*math.pi/num_directions)
        v = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)
        self.wdect = WDECTLayer(config, v)
        in_dim     = num_directions * config.bump_steps
        self.fc1   = nn.Linear(in_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)
        flat  = wfeat.view(batch.num_graphs, -1)
        x     = F.relu(self.fc1(flat))
        return self.fc2(x), flat

    def compute_loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# -----------------------------------------------------------------------------
# 3) Train & eval
# -----------------------------------------------------------------------------
def train(model, data_list, epochs=100, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        batch = Batch.from_data_list(data_list)
        logits, feats = model(batch)
        loss = model.compute_loss(logits, batch.y, feats)
        loss.backward()
        opt.step()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs}, Loss: {loss:.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        return (pred == batch.y).float().mean().item()

# -----------------------------------------------------------------------------
# 4) Run all four
# -----------------------------------------------------------------------------
if __name__=="__main__":
    config      = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    num_dirs    = 64
    hidden_dim  = 64

    datasets = [
        ("Digits",      load_digits_data,       10),
        ("Iris",        load_iris_data,         3),
        ("BreastCancer",load_breast_cancer_data,2),
        ("Letters",     load_letters_data,      26),
    ]

    for name, loader, n_cls in datasets:
        print(f"\n=== {name} ===")
        data_list      = loader()
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

        model = WDECTClassifier(config, hidden_dim, n_cls, num_dirs)
        train(model, train_data, epochs=200)
        print(" Train Acc:", evaluate(model, train_data))
        print(" Test  Acc:", evaluate(model, test_data))
