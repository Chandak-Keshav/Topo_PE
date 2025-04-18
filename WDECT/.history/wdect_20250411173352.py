"""
wdect_demo.py

This script implements the Weighted Differentiable Euler Characteristic Transform (WDECT)
using all simplices (vertices, edges, faces) on the load_digits dataset.

Required modules:
    - build_weighted_complex (from build_weighted_complex.py)
    - complex_to_weighted_ECT, weighted_euler_curve (from complex_to_weighted_ECT.py)
    - distance_RotationInvariant (from distance_RotationInvariant.py)
    - ECTConfig, Batch, normalize (from ect.py)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Pre-existing imports as stated:
from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT, weighted_euler_curve
from distance_RotationInvariant import distance_RotationInvariant
from ect import ECTConfig, Batch, normalize

### UTILITY FUNCTIONS ###

def sample_directions(num_directions):
    """
    Sample `num_directions` uniformly over the unit circle.
    Returns:
        directions (np.ndarray): Array of shape (num_directions, 2)
    """
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return directions

def compute_wecc(values, weights, thresholds, lam):
    """
    Computes the weighted Euler characteristic curve contribution given projection values
    and corresponding weights. For each threshold r, compute:
        sum_i weights[i] * sigmoid( lam * (r - values[i]) )
    Vectorized over all thresholds.

    Parameters:
        values (np.ndarray): Array of shape (N, ) with projection values.
        weights (np.ndarray): Array of shape (N, ) with corresponding weights.
        thresholds (np.ndarray): Array of threshold values.
        lam (float): Sharpness parameter for sigmoid.

    Returns:
        np.ndarray: Array of shape (len(thresholds),) with the weighted sum.
    """
    # values: shape (N,1); thresholds: shape (1, T)
    diff = thresholds[None, :] - values[:, None]
    sig = 1.0 / (1.0 + np.exp(-lam * diff))
    return np.sum(weights[:, None] * sig, axis=0)

def complex_to_weighted_DECT(complex_data, num_directions=16, num_steps=16, lam=10.0):
    """
    Compute the Weighted DECT representation using all simplices (points, edges, faces).

    Parameters:
        complex_data (dict): A dictionary containing the weighted complex with keys:
            'V'         : vertices, shape (num_V, 2)
            'E'         : edges, shape (num_edges, 2) containing indices into V
            'F'         : faces, shape (num_faces, 3) containing indices into V
            'V_weights' : vertex weights, shape (num_V,)
            'E_weights' : edge weights, shape (num_edges,)
            'F_weights' : face weights, shape (num_faces,)
        num_directions (int): Number of directions to sample.
        num_steps (int): Number of threshold discretization steps.
        lam (float): Sigmoid sharpness parameter.

    Returns:
        np.ndarray: A matrix of shape (num_directions, num_steps) representing the WDECT.
    """
    directions = sample_directions(num_directions)
    # Determine overall projection range by aggregating over vertices
    all_proj = []
    V = complex_data['V']  # shape (num_V, 2)
    for d in directions:
        all_proj.extend((V @ d).tolist())
    all_proj = np.array(all_proj)
    thresholds = np.linspace(np.min(all_proj), np.max(all_proj), num_steps)

    wdect = np.zeros((num_directions, num_steps))
    # For each direction, compute contributions from vertices, edges, and faces.
    for i, d in enumerate(directions):
        # --- POINTS (Vertices) ---
        V_proj = V @ d  # projection of each vertex
        points_contrib = compute_wecc(V_proj, complex_data['V_weights'], thresholds, lam)
        
        # --- EDGES ---
        if complex_data['E'].size > 0:
            # For each edge (defined by two vertex indices), compute min projection
            E_indices = complex_data['E']  # shape (num_E, 2)
            E_proj = np.minimum(V_proj[E_indices[:, 0]], V_proj[E_indices[:, 1]])
            edges_contrib = compute_wecc(E_proj, complex_data['E_weights'], thresholds, lam)
        else:
            edges_contrib = 0

        # --- FACES ---
        if complex_data['F'].size > 0:
            # For each face (triangle), compute min projection over its 3 vertices.
            F_indices = complex_data['F']  # shape (num_F, 3)
            F_proj = np.min(np.stack([V_proj[F_indices[:, 0]], V_proj[F_indices[:, 1]], V_proj[F_indices[:, 2]]], axis=1), axis=1)
            faces_contrib = compute_wecc(F_proj, complex_data['F_weights'], thresholds, lam)
        else:
            faces_contrib = 0

        # Combine contributions according to alternating sum:
        # Weighted Euler characteristic = (points) - (edges) + (faces)
        wdect[i, :] = points_contrib - edges_contrib + faces_contrib

    return wdect

### MAIN PIPELINE ###

if __name__ == '__main__':
    # Load the load_digits dataset from scikit-learn
    digits = load_digits()
    images = digits.images       # Each image: shape (height, width)
    labels = digits.target       # Labels: digits 0-9

    all_features = []
    all_labels = []
    
    print("Processing images to build weighted complexes and compute WDECT representations...")
    # Process each image to build the weighted complex and compute its WDECT.
    for idx, img in enumerate(images):
        # Build weighted complex from image.
        # The build_weighted_complex function should return:
        # V, E, F, V_weights, E_weights, F_weights.
        V, E, F, V_weights, E_weights, F_weights = build_weighted_complex(img)
        complex_data = {
            'V': V,
            'E': E,
            'F': F,
            'V_weights': V_weights,
            'E_weights': E_weights,
            'F_weights': F_weights,
        }
        wdect_rep = complex_to_weighted_DECT(complex_data, num_directions=16, num_steps=16, lam=10.0)
        # Optionally, you may smooth or normalize wdect_rep here.
        # Flatten the 2D WDECT representation into a feature vector.
        all_features.append(wdect_rep.flatten())
        all_labels.append(labels[idx])
        if idx % 100 == 0:
            print(f"Processed image {idx}")

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Visualize an example WDECT representation
    example_rep = all_features[0].reshape(16, 16)
    plt.figure(figsize=(6, 4))
    plt.imshow(example_rep, aspect='auto', cmap='viridis')
    plt.xlabel("Threshold steps")
    plt.ylabel("Projection directions")
    plt.colorbar()
    plt.title("WDECT Representation (Example)")
    plt.show()

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels,
                                                        test_size=0.3, random_state=42)
    
    # Train a simple 1-NN classifier on the WDECT features.
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot the confusion matrix.
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='viridis')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
