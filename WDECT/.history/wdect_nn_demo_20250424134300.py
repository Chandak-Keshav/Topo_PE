"""
wdect_nn_demo.py

This script demonstrates a neural network classifier that integrates the 
Weighted Differentiable Euler Characteristic Transform (WDECT) into its architecture.
It relies on weighted complex construction (via build_weighted_complex) and weighted ECT 
computation (via complex_to_weighted_ECT and wect.py) and uses the load_digits dataset 
for training and evaluation.

Required modules:
    - build_weighted_complex (from build_weighted_complex.py)
    - complex_to_weighted_ECT, weighted_euler_curve (from complex_to_weighted_ECT.py)
    - distance_RotationInvariant (from distance_RotationInvariant.py)
    - ECTConfig, Batch, normalize (from ect.py)
    - WECTLayer (from wect.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import Data, Batch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Pre-existing helper modules (assumed implemented)
from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT, weighted_euler_curve
from distance_RotationInvariant import distance_RotationInvariant
from ect import ECTConfig, Batch as ECTBatch, normalize
from wect import ECTLayer

################################################################################
# Data Preparation: Build Weighted Complex Data for Each Image
################################################################################
def load_digits_weighted_complex_data():
    """
    Loads the load_digits dataset and converts each image into a weighted complex.
    The build_weighted_complex(image) function is assumed to return:
        V, E, F, V_weights, E_weights, F_weights
    where:
        - V: vertices, np.array of shape (num_V, 2)
        - E: edges, np.array of shape (num_edges, 2) -- indices into V
        - F: faces, np.array of shape (num_faces, 3) -- indices into V
        - V_weights: vertex weights, np.array of shape (num_V,)
        - E_weights: edge weights, np.array of shape (num_edges,)
        - F_weights: face weights, np.array of shape (num_faces,)
    Returns:
        data_list: List of torch_geometric.data.Data objects with attributes:
                   x, edge_index, face, node_weights, and label y.
    """
    digits = load_digits()
    images = digits.images      # shape: (n_samples, height, width)
    targets = digits.target     # labels: digits 0-9
    
    data_list = []
    for i in range(len(images)):
        img = images[i]
        V, E, F, V_weights, E_weights, F_weights = build_weighted_complex(img)
        # Convert vertices and weights into torch tensors
        x = torch.tensor(V, dtype=torch.float)              # Coordinates, shape (num_V, 2)
        node_weights = torch.tensor(V_weights, dtype=torch.float)  # Vertex weights, shape (num_V,)
        # Process edge indices: ensure shape is [2, num_edges]
        if E.size:
            edge_index = torch.tensor(E.T, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        # Process face indices: ensure shape is [3, num_faces]
        if F.size:
            face = torch.tensor(F.T, dtype=torch.long)
        else:
            face = torch.empty((3, 0), dtype=torch.long)
        data_item = Data(x=x, edge_index=edge_index, face=face, node_weights=node_weights,
                         y=torch.tensor([targets[i]], dtype=torch.long))
        data_list.append(data_item)
    return data_list

################################################################################
# Weighted DECT (WDECT) Classifier Definition
################################################################################
class WECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int):
        """
        Initializes the classifier that uses the Weighted ECT (WECT) layer to compute
        a topological signature from each weighted complex.
        
        Parameters:
            config: An instance of ECTConfig with parameters for the WECT layer.
            hidden_dim: Dimension of the hidden fully connected layer.
            num_classes: Number of output classes.
        """
        super(WECTClassifier, self).__init__()
        # Use the WECTLayer (which is built on weighted logic) from wect.py
        self.wect_layer = WECTLayer(config)
        # The WECTLayer is expected to output a tensor of shape 
        # [batch_size, num_directions, bump_steps]. We flatten this output.
        self.fc1 = nn.Linear(config.num_directions * config.bump_steps, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Data):
        """
        Forward pass:
            - Compute the weighted topological signature using the WECTLayer.
            - Flatten the output and pass it through a two-layer MLP.
        """
        # Compute the WECT (weighted DECT) representation.
        wect_out = self.wect_layer(batch)
        # If the output has more than 2 dimensions, flatten it.
        if wect_out.dim() > 2:
            wect_feat = wect_out.view(wect_out.size(0), -1)
        else:
            wect_feat = wect_out
        x = F.relu(self.fc1(wect_feat))
        x = self.fc2(x)
        return x

################################################################################
# Training and Evaluation Functions
################################################################################
def train(model, train_data, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        batch = Batch.from_data_list(train_data)
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

def evaluate(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        out = model(batch)
        pred = out.argmax(dim=1)
        correct = (pred == batch.y).sum().item()
        accuracy = correct / batch.num_graphs
    return accuracy

################################################################################
# Main Execution
################################################################################
def main():
    # Set up the ECT/WECT configuration.
    # Adjust these parameters as needed. The following are sample values.
    config = ECTConfig(
        bump_steps=32,          # Number of threshold steps (discretization resolution)
        radius=1.1,             # Radius for the filtration
        ect_type="faces",       # Set to "points", "edges", or "faces" (or a combination via alternating sums)
        normalized=True,        # Normalize the computed topological signature
        fixed=True,             # Whether to keep the directions fixed during training
        # Below, num_directions is an extra parameter added in your config for clarity.
        # (If not already present, you can add it to your ECTConfig dataclass.)
    )
    # For this example, assume config includes num_directions as an attribute.
    config.num_directions = 128  # e.g., 128 projection directions

    hidden_dim = 64
    num_classes = 10    # For digits 0-9
    num_epochs = 200

    # Load and convert the digits dataset into weighted complex Data objects.
    data_list = load_digits_weighted_complex_data()

    # Split into training and testing sets.
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

    # Initialize the WECT classifier.
    model = WECTClassifier(config, hidden_dim, num_classes)

    # Train the model.
    train(model, train_data, num_epochs=num_epochs)

    # Evaluate on training and testing sets.
    train_acc = evaluate(model, train_data)
    test_acc = evaluate(model, test_data)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Optional: Visualize one WECT representation from the test batch.
    batch = Batch.from_data_list(test_data)
    wect_out = model.wect_layer(batch)
    # Reshape the first sample's signature for display.
    sample_rep = wect_out[0].cpu().detach().numpy().reshape(config.num_directions, config.bump_steps)
    plt.figure(figsize=(6, 4))
    plt.imshow(sample_rep, aspect='auto', cmap='viridis')
    plt.xlabel("Threshold Steps")
    plt.ylabel("Projection Directions")
    plt.title("Sample WDECT Representation")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
