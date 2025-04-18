"""
wdect_nn_demo.py

This script demonstrates a neural network classifier that
integrates a Weighted Differentiable Euler Characteristic Transform (WDECT)
layer. It uses the load_digits dataset, converts each image into a weighted complex,
and classifies the images using a network built on top of the WECT layer.

Required modules:
    - build_weighted_complex (from build_weighted_complex.py)
    - complex_to_weighted_ECT, weighted_euler_curve (from complex_to_weighted_ECT.py)
    - distance_RotationInvariant (from distance_RotationInvariant.py)
    - ECTConfig, Batch, normalize, and WECTLayer (from ect.py and wect.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-existing imports (assumed to be implemented)
from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT, weighted_euler_curve
from distance_RotationInvariant import distance_RotationInvariant
from ect import ECTConfig, Batch as ECTBatch, normalize
from wect import WECTLayer

################################################################################
# Helper: Build a torch-geometric Data object for each image using the weighted complex.
################################################################################
def load_digits_weighted_complex_data():
    """
    Loads the load_digits dataset and converts each image into a weighted complex.
    
    Assumes build_weighted_complex(image) returns:
        V, E, F, V_weights, E_weights, F_weights
    where:
        - V: vertices, np.array of shape (num_V, 2)
        - E: edges, np.array of shape (num_edges, 2) -- indices into V
        - F: faces, np.array of shape (num_faces, 3) -- indices into V
        - V_weights: vertex weights, np.array of shape (num_V,)
        - E_weights: edge weights, np.array of shape (num_edges,)
        - F_weights: face weights, np.array of shape (num_faces,)
    
    Returns:
        data_list: list of torch_geometric.data.Data objects with attributes:
                   x, node_weights, edge_index, face, y.
    """
    data = load_digits()
    images = data.images   # images: (n_samples, height, width)
    labels = data.target   # labels: digits 0-9
    
    data_list = []
    for i in range(len(images)):
        img = images[i]
        # Build the weighted complex for the image
        V, E, F, V_weights, E_weights, F_weights = build_weighted_complex(img)
        # Convert to torch tensors
        x = torch.tensor(V, dtype=torch.float)                  # vertex coordinates, shape (num_V, 2)
        node_weights = torch.tensor(V_weights, dtype=torch.float) # vertex weights, shape (num_V,)
        # For edges, transpose to shape (2, num_edges) if not empty
        if E.size:
            edge_index = torch.tensor(E.T, dtype=torch.long)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
        # For faces, transpose to shape (3, num_faces) if not empty
        if F.size:
            face = torch.tensor(F.T, dtype=torch.long)
        else:
            face = torch.empty((3,0), dtype=torch.long)
        
        data_item = Data(x=x, node_weights=node_weights, edge_index=edge_index, face=face,
                         y=torch.tensor([labels[i]], dtype=torch.long))
        data_list.append(data_item)
    return data_list

################################################################################
# Define the Weighted DECT Classifier using the WECTLayer.
################################################################################
class WECTClassifier(nn.Module):
    def __init__(self, config: ECTConfig, hidden_dim: int, num_classes: int):
        """
        Initializes the classifier.
        
        Parameters:
            config: ECTConfig object containing parameters for the WECT layer.
            hidden_dim: Dimension of the hidden fully connected layer.
            num_classes: Number of output classes.
        """
        super(WECTClassifier, self).__init__()
        # Initialize WECTLayer from your wect.py implementation.
        # This layer will compute the Weighted Euler Characteristic Transform (WECT)
        # for each graph (weighted complex) in the batch.
        self.wect_layer = WECTLayer(config)
        # The output shape of WECTLayer is typically [batch, num_directions, bump_steps].
        # We flatten that output to feed into a fully connected network.
        self.fc1 = nn.Linear(config.num_directions * config.bump_steps, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: Data):
        """
        Forward pass:
            - Compute the WECT representation using the provided WECTLayer.
            - Flatten the representation and run through FC layers.
        """
        # Compute the weighted ECT (WDECT) using the WECTLayer.
        # The layer expects a Batch object, so we assume `batch` comes from torch_geometric.
        wect_out = self.wect_layer(batch)
        # Ensure the output is of shape [batch_size, num_directions, bump_steps]
        if wect_out.dim() > 2:
            wect_feat = wect_out.view(wect_out.size(0), -1)
        else:
            wect_feat = wect_out
        # Pass through the MLP
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
    # Set up configuration for the WECT layer.
    # Adjust these parameters as needed; for example:
    #   - radius: scaling for the filtration (e.g., 1.0)
    #   - bump_steps: number of discrete threshold steps (e.g., 16)
    #   - num_directions: number of projection directions (e.g., 128)
    #   - ect_type: type of simplices used ("points", "edges", or "faces"). You can also combine them using alternating sums.
    config = ECTConfig(
        radius=1.0,
        bump_steps=16,
        num_directions=128,  # number of directions used in the WECT.
        ect_type="faces",    # choose "points", "edges", or "faces". For full weighted DECT, you might combine them.
        normalized=True,
        fixed=True
    )

    hidden_dim = 64
    num_classes = 10  # digits 0-9
    num_epochs = 200

    # Load and process data into weighted complexes.
    data_list = load_digits_weighted_complex_data()
    
    # Split into training and testing sets
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

    # Initialize and train the WECT classifier.
    model = WECTClassifier(config, hidden_dim, num_classes)
    train(model, train_data, num_epochs=num_epochs)

    # Evaluate the model.
    train_acc = evaluate(model, train_data)
    test_acc = evaluate(model, test_data)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # (Optional) Visualize one WECT representation from the batch.
    batch = Batch.from_data_list(test_data)
    wect_out = model.wect_layer(batch)
    # Take the first sample from the batch and reshape its representation.
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
