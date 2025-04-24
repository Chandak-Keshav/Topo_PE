import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# -------------- your weighted ECT lib (unchanged) --------------
from wdect import ECTConfig, ECTLayer

# 1) Load & preprocess Letters
def load_letters():
    D = fetch_openml('letter', version=1)
    X = D.data.to_numpy()
    y = LabelEncoder().fit_transform(D.target.to_numpy())
    return X, y

X, y = load_letters()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2) Build Data + Batch with default weights & y
def make_batch(X, y):
    data_list = []
    for xi, yi in zip(X, y):
        xi = torch.tensor(xi, dtype=torch.float).unsqueeze(0)  # [1,16]
        data_list.append(Data(
            x            = xi,
            y            = torch.tensor([yi], dtype=torch.long),
            node_weights = torch.ones(xi.size(0)),
            edge_index   = torch.empty((2,0), dtype=torch.long),
            face         = torch.empty((3,0), dtype=torch.long),
            edge_weights = torch.empty((0,)),
            face_weights = torch.empty((0,))
        ))
    return Batch.from_data_list(data_list)

train_batch = make_batch(X_train, y_train)
test_batch  = make_batch(X_test,  y_test)

# 3) Instantiate weighted + differentiable ECT extractor
DEVICE     = "cpu"
num_feats  = X_train.shape[1]   # 16
num_thetas = 128
# random directions in R^16
v = torch.randn(num_feats, num_thetas)

config = ECTConfig(
    ect_type   = "points",
    bump_steps = 128,
    radius     = 1.0,
    normalized = False,
    fixed      = True
)
ect_layer = ECTLayer(config, v=v).to(DEVICE)

def extract_ect(batch):
    out = ect_layer(batch.to(DEVICE))              # [N, num_thetas, bump_steps]
    N   = len(batch.y)                              # now exists
    return out.cpu().numpy().reshape(N, -1)         # [N, num_thetas*bump_steps]

E_train = extract_ect(train_batch)
E_test  = extract_ect(test_batch)

# 4) SVM on ECT features
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(E_train, y_train)
y_pred = svm.predict(E_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 5) Optional: visualize first two features
plt.figure(figsize=(6,4))
plt.scatter(E_test[:,0], E_test[:,1], c=y_pred, cmap="tab20", s=10)
plt.title("Letters via Weighted‑Differentiable ECT")
plt.xlabel("ECT Feature 1")
plt.ylabel("ECT Feature 2")
plt.colorbar(label="Predicted Class")
plt.show()
