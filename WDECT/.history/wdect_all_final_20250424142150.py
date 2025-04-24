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
# WDECTLayer: Weighted + Differentiable
# -----------------------------------------------------------------------------
class WDECTLayer(nn.Module):
    def __init__(self, config: ECTConfig, v: torch.Tensor):
        super().__init__()
        self.config = config
        self.lin = nn.Parameter(
            torch.linspace(-config.radius, config.radius, config.bump_steps).view(-1,1,1,1),
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
        return compute_ecc(nh, batch.batch, lin) - compute_ecc(eh, idx_e, lin) + compute_ecc(fh, idx_f, lin)

# -----------------------------------------------------------------------------
# Helper function to create a complete graph from a feature vector
# -----------------------------------------------------------------------------
def create_complete_graph(f: torch.Tensor) -> Data:
    """Convert a feature vector into a complete graph for WDECTLayer."""
    F = len(f)
    # Nodes: F nodes, each at position e_i * f[i] in F-dimensional space
    x = torch.zeros(F, F)
    for i in range(F):
        x[i, i] = f[i]
    edge_index = torch.combinations(torch.arange(F), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    face = torch.combinations(torch.arange(F), r=3).T if F >= 3 else torch.empty((3,0), dtype=torch.long)
    node_weights = torch.ones(F)
    edge_weights = torch.ones(edge_index.shape[1])
    face_weights = torch.ones(face.shape[1]) if face.numel() > 0 else torch.empty(0)
    return Data(x=x, edge_index=edge_index, face=face, node_weights=node_weights, 
                edge_weights=edge_weights, face_weights=face_weights)

# -----------------------------------------------------------------------------
# Modified WDECTClassifier with flexible input dimension
# -----------------------------------------------------------------------------
class WDECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int, num_directions: int, dim: int):
        super().__init__()
        self.config = config
        # Generate random unit directions in R^dim
        v = torch.randn(dim, num_directions)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        in_dim = num_directions * config.bump_steps
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Batch):
        wfeat = self.wdect(batch)  # [B, D, S]
        flat = wfeat.view(batch.num_graphs, -1)
        x = F.relu(self.fc1(flat))
        return self.fc2(x), flat

    def compute_loss(self, logits, targets, feats, λ=0.1):
        return F.cross_entropy(logits, targets) + λ * torch.norm(feats, p=2)

# -----------------------------------------------------------------------------
# Data loader for all datasets
# -----------------------------------------------------------------------------
def load_and_prepare_dataset(dataset_name: str):
    """Load dataset and convert samples to graph structures."""
    if dataset_name == "load_digits":
        data = load_digits()
        imgs, labels = data.images, data.target
        data_list = []
        for img, lbl in zip(imgs, labels):
            V, E, F, Vw, Ew, Fw = build_weighted_complex(img)
            data_obj = Data(
                x=torch.tensor(V, dtype=torch.float),
                edge_index=torch.tensor(E.T, dtype=torch.long) if E.size else torch.empty((2,0), dtype=torch.long),
                face=torch.tensor(F.T, dtype=torch.long) if F.size else torch.empty((3,0), dtype=torch.long),
                node_weights=torch.tensor(Vw, dtype=torch.float),
                edge_weights=torch.tensor(Ew, dtype=torch.float),
                face_weights=torch.tensor(Fw, dtype=torch.float),
                y=torch.tensor([lbl], dtype=torch.long)
            )
            data_list.append(data_obj)
        dim = 2  # 2D coordinates from image
        num_classes = 10
    elif dataset_name == "iris":
        data = load_iris()
        features = torch.tensor(data.data, dtype=torch.float)
        labels = torch.tensor(data.target, dtype=torch.long)
        data_list = [create_complete_graph(f).clone() for f in features]
        for data_obj, lbl in zip(data_list, labels):
            data_obj.y = lbl.unsqueeze(0)
        dim = features.shape[1]  # 4 features
        num_classes = 3
    elif dataset_name == "spect":
        data = fetch_openml(name='SPECT', version=1, parser='auto')
        # Convert categorical columns to numeric using cat.codes
        data.data = data.data.apply(lambda x: x.cat.codes if pd.api.types.is_categorical_dtype(x) else x)
        # Now convert to tensor
        features = torch.tensor(data.data.to_numpy(), dtype=torch.float)
        labels = torch.tensor((data.target == '1').astype(int), dtype=torch.long)
        data_list = [create_complete_graph(f).clone() for f in features]
        for data_obj, lbl in zip(data_list, labels):
            data_obj.y = lbl.unsqueeze(0)
        dim = features.shape[1]  # 22 features
        num_classes = 2
    elif dataset_name == "letters":
        data = fetch_openml(name='letter', version=1, parser='auto')
        features = torch.tensor(data.data.to_numpy(), dtype=torch.float)
        labels = torch.tensor(data.target.map(lambda x: ord(x) - ord('A')), dtype=torch.long)
        data_list = [create_complete_graph(f).clone() for f in features]
        for data_obj, lbl in zip(data_list, labels):
            data_obj.y = lbl.unsqueeze(0)
        dim = features.shape[1]  # 16 features
        num_classes = 26
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return data_list, dim, num_classes

# -----------------------------------------------------------------------------
# Train and Evaluate
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
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        return (pred == batch.y).float().mean().item()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    datasets = ["iris", "spect", "load_digits", "letters"]
    config = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    num_directions, hidden_dim = 64, 64

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} dataset")
        data_list, dim, num_classes = load_and_prepare_dataset(dataset_name)
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
        model = WDECTClassifier(config, hidden_dim, num_classes, num_directions, dim)
        train(model, train_data, epochs=100)
        train_acc = evaluate(model, train_data)
        test_acc = evaluate(model, test_data)
        print(f"{dataset_name} Train Accuracy: {train_acc:.4f}")
        print(f"{dataset_name} Test Accuracy: {test_acc:.4f}")