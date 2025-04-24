import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_wine, load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import train_test_split
from wect import ECTConfig, compute_ecc, normalize
from build_weighted_complex import build_weighted_complex
import pandas as pd
import numpy as np

# WDECTLayer: Weighted + Differentiable
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

# Helper function to create a complete graph from a feature vector
def create_complete_graph(f: torch.Tensor) -> Data:
    """Convert a feature vector into a complete graph for WDECTLayer."""
    F = len(f)
    # Nodes: F nodes, each at position e_i * f[i] in F-dimensional space
    x = torch.zeros(F, F)
    for i in range(F):
        x[i, i] = f[i]
    # Edges: all possible pairs (undirected)
    edge_index = torch.combinations(torch.arange(F), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Faces: all possible triangles
    face = torch.combinations(torch.arange(F), r=3).T if F >= 3 else torch.empty((3,0), dtype=torch.long)
    # Weights: all set to 1 for simplicity
    node_weights = torch.ones(F)
    edge_weights = torch.ones(edge_index.shape[1])
    face_weights = torch.ones(face.shape[1]) if face.numel() > 0 else torch.empty(0)
    return Data(x=x, edge_index=edge_index, face=face, node_weights=node_weights, 
                edge_weights=edge_weights, face_weights=face_weights)

# Modified WDECTClassifier with flexible input dimension
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

# Data loader for all datasets
def load_and_prepare_dataset(dataset_name: str):
    """Load dataset and convert samples to graph structures."""
    if dataset_name == "wine":
        data = load_wine()
        features = torch.tensor(data.data, dtype=torch.float)
        labels = torch.tensor(data.target, dtype=torch.long)
        dim = features.shape[1]  # 13
        num_classes = len(set(data.target))  # 3
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        features = torch.tensor(data.data, dtype=torch.float)
        labels = torch.tensor(data.target, dtype=torch.long)
        dim = features.shape[1]  # 30
        num_classes = len(set(data.target))  # 2
    elif dataset_name == "congress":
        # Load data from UCI repository
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'
        data = pd.read_csv(url, header=None)
        # Map categorical features to numerical values
        mapping = {'y': 1, 'n': 0, '?': -1}
        features = data.iloc[:, 1:].applymap(lambda x: mapping[x])
        labels = data.iloc[:, 0].map({'democrat': 0, 'republican': 1})
        # Convert to PyTorch tensors
        features = torch.tensor(features.values, dtype=torch.float)
        labels = torch.tensor(labels.values, dtype=torch.long)
        dim = features.shape[1]  # 16
        num_classes = 2
    elif dataset_name.startswith("moons_"):
        parts = dataset_name.split("_")
        size = int(parts[1])
        noise = float(parts[2])
        X, y = make_moons(n_samples=size, noise=noise, random_state=42)
        features = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(y, dtype=torch.long)
        dim = 2
        num_classes = 2
    elif dataset_name.startswith("circles_"):
        parts = dataset_name.split("_")
        size = int(parts[1])
        noise = float(parts[2])
        X, y = make_circles(n_samples=size, noise=noise, factor=0.5, random_state=42)
        features = torch.tensor(X, dtype=torch.float)
        labels = torch.tensor(y, dtype=torch.long)
        dim = 2
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data_list = [create_complete_graph(f).clone() for f in features]
    for data_obj, lbl in zip(data_list, labels):
        data_obj.y = lbl.unsqueeze(0)
    return data_list, dim, num_classes

# Train and Evaluate
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

# Main
if __name__ == "__main__":
    # Define dataset variations
    sizes = [1000, 5000, 10000, 20000]
    noises = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    moons_datasets = [f"moons_{size}_{noise}" for size in sizes for noise in noises]
    circles_datasets = [f"circles_{size}_{noise}" for size in sizes for noise in noises]
    datasets = ["wine", "congress", "breast_cancer"] + moons_datasets + circles_datasets
    
    config = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    num_directions, hidden_dim = 64, 64

    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name} dataset")
        data_list, dim, num_classes = load_and_prepare_dataset(dataset_name)
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
        model = WDECTClassifier(config, hidden_dim, num_classes, num_directions, dim)
        train(model, train_data, epochs=300)
        train_acc = evaluate(model, train_data)
        test_acc = evaluate(model, test_data)
        print(f"{dataset_name} Train Accuracy: {train_acc:.4f}")
        print(f"{dataset_name} Test Accuracy: {test_acc:.4f}")