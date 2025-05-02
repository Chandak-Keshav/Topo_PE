import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wect import ECTConfig, compute_ecc, normalize
from build_weighted_complex import build_weighted_complex
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Graph creation for vector features
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

# WDECTLayer (with empty-face guard)
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

# RFF for RBF-kernel approximation
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

# SVM-like classifier module
class WDECT_RBFSVM(nn.Module):
    def __init__(self, config: ECTConfig, dim, num_directions, rff_dim, gamma, num_classes):
        super().__init__()
        v = torch.randn(dim, num_directions, device=device)
        v = v / v.norm(dim=0, keepdim=True)
        self.wdect = WDECTLayer(config, v)
        feat_dim = num_directions * config.bump_steps
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

# Data loader for all datasets
def load_and_prepare_dataset(name: str, scaler: StandardScaler):
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
        dim, num_classes, gamma = 2, 10, 0.5
    elif name == "iris":
        data = load_iris()
        feats = scaler.fit_transform(data.data)
        feats = torch.tensor(feats, dtype=torch.float, device=device)
        labs = torch.tensor(data.target, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes, gamma = feats.shape[1], 3, 0.1
    elif name == "spect":
        url = 'https://archive.ics.uci.edu/static/public/95/spect+heart.zip'
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download SPECT dataset from {url}")
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        train_data = pd.read_csv(zip_file.open('SPECT.train'), header=None)
        test_data = pd.read_csv(zip_file.open('SPECT.test'), header=None)
        data = pd.concat([train_data, test_data], ignore_index=True)
        feats = scaler.fit_transform(data.iloc[:, 1:].values)
        feats = torch.tensor(feats, dtype=torch.float, device=device)
        labs = torch.tensor(data.iloc[:, 0].values, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes, gamma = feats.shape[1], 2, 1.0
    elif name == "wine":
        data = load_wine()
        feats = scaler.fit_transform(data.data)
        feats = torch.tensor(feats, dtype=torch.float, device=device)
        labs = torch.tensor(data.target, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes, gamma = feats.shape[1], 3, 0.5
    elif name == "breast_cancer":
        data = load_breast_cancer()
        feats = scaler.fit_transform(data.data)
        feats = torch.tensor(feats, dtype=torch.float, device=device)
        labs = torch.tensor(data.target, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes, gamma = feats.shape[1], 2, 0.5
    elif name == "congress":
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'
        data = pd.read_csv(url, header=None)
        mapping = {'y': 1, 'n': 0, '?': -1}
        feats = data.iloc[:, 1:].applymap(lambda x: mapping[x])
        feats = scaler.fit_transform(feats.values)
        feats = torch.tensor(feats, dtype=torch.float, device=device)
        labs = torch.tensor(data.iloc[:, 0].map({'democrat': 0, 'republican': 1}).values, dtype=torch.long, device=device)
        data_list = []
        for f, l in zip(feats, labs):
            g = create_complete_graph(f)
            g.y = l.unsqueeze(0)
            data_list.append(g)
        dim, num_classes, gamma = feats.shape[1], 2, 0.5
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return data_list, dim, num_classes, gamma

# Training & evaluation helpers
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

    for ep in range(1, epochs+1):
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

        if ep % 10 == 0:
            print(f"Epoch {ep:02d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep}")
            model.load_state_dict(best_model_state)
            break

# Main
if __name__ == "__main__":
    datasets = ["iris", "spect", "load_digits", "wine", "breast_cancer", "congress"]
    config = ECTConfig(radius=1.0, bump_steps=16, ect_type="faces", normalized=True, fixed=True)
    num_dirs, rff_dim = 32, 256
    lr, epochs = 1e-2, 200
    scaler = StandardScaler()

    accuracies = []
    for name in datasets:
        print(f"\n--- {name} ---")
        data_list, dim, num_cls, gamma = load_and_prepare_dataset(name, scaler)
        train_val, test = train_test_split(data_list, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        model = WDECT_RBFSVM(config, dim, num_dirs, rff_dim, gamma, num_cls).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
        
        train_with_early_stopping(model, train, val, opt, scheduler, epochs)
        
        tr_acc = evaluate(model, train)
        te_acc = evaluate(model, test)
        print(f"Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")
        accuracies.append(f"Dataset: {name}, Train Acc: {tr_acc:.4f}, Test Acc: {te_acc:.4f}")

    # Save accuracies to file
    with open("accuracies.txt", "w") as f:
        for acc in accuracies:
            f.write(acc + "\n")