import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from ucimlrepo import fetch_ucirepo
import pandas as pd

from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT
num_directions = 10     
num_steps = 20          
method = 'gaussian'      
window = 0.2 * num_steps 
normalization_method = 'none'  
lambda_ = 1.0           
C_svm = 1.0             


def compute_distance_grid(X, indices, i):
    """
    Compute a normalized distance matrix for a point's k-nearest neighbors.
    
    Args:
        X (np.ndarray): Data matrix (n_samples, n_features)
        indices (np.ndarray): Indices of k-nearest neighbors for each point
        i (int): Index of the current point
    
    Returns:
        np.ndarray: Normalized k × k distance matrix
    """
    neighbors = X[indices[i]]
    dist_matrix = cdist(neighbors, neighbors)
    min_dist = dist_matrix.min()
    max_dist = dist_matrix.max()
    if max_dist > min_dist:
        dist_matrix = (dist_matrix - min_dist) / (max_dist - min_dist)
    else:
        dist_matrix = np.zeros_like(dist_matrix)
    return dist_matrix

def compute_wect_features(X, k):
    """
    Compute WECT features for all points in the dataset.
    
    Args:
        X (np.ndarray): Data matrix (n_samples, n_features)
        k (int): Number of nearest neighbors
    
    Returns:
        np.ndarray: WECT feature matrix (n_samples, feature_dim)
    """
    n_samples = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(X)
    wect_features = []
    for i in range(n_samples):
        grid = compute_distance_grid(X, indices, i)
        V, E, F, V_w, E_w, F_w = build_weighted_complex(grid)
        complex = {'V': V, 'E': E, 'F': F, 'V_weights': V_w, 'E_weights': E_w, 'F_weights': F_w}
        wect = complex_to_weighted_ECT(complex, num_directions, num_steps, method, window, normalization_method)
        wect_features.append(wect.flatten())
        if i % 100 == 0:
            print(f"Computed WECT for point {i}/{n_samples}")
    return np.array(wect_features)

def process_dataset(X, y, dataset_name):
    """
    Process a dataset: compute WECT, train SVM with combined kernel, and evaluate.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        dataset_name (str): Name of the dataset
    
    Returns:
        float: Test accuracy
    """
    n_samples = X.shape[0]
    k = min(50, n_samples // 10)  
    print(f"Processing {dataset_name} with {n_samples} samples, k={k}")

    wect_features = compute_wect_features(X, k)
 
    X_train, X_test, y_train, y_test, wect_train, wect_test = train_test_split(
        X, y, wect_features, test_size=0.2, random_state=42
    )
  
    gamma_rbf = 1 / (X_train.shape[1] * X_train.var())
    K_rbf_train = rbf_kernel(X_train, X_train, gamma=gamma_rbf)
    K_rbf_test_train = rbf_kernel(X_test, X_train, gamma=gamma_rbf)
 
    gamma_wect = 1 / np.mean(cdist(wect_train, wect_train))
    K_wect_train = np.exp(-gamma_wect * cdist(wect_train, wect_train)**2)
    K_wect_test_train = np.exp(-gamma_wect * cdist(wect_test, wect_train)**2)
    
    K_total_train = K_rbf_train + lambda_ * K_wect_train
    K_total_test_train = K_rbf_test_train + lambda_ * K_wect_test_train
    
    svm = SVC(kernel='precomputed', C=C_svm)
    svm.fit(K_total_train, y_train)
   
    y_pred = svm.predict(K_total_test_train)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy for {dataset_name}: {accuracy:.4f}")
    return accuracy



def load_spect_data():
    """
    Load the SPECT Heart dataset from UCI (ID 95).
    
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    try:
        spect = fetch_ucirepo(id=95)  
        X = spect.data.features.to_numpy()
        y = spect.data.targets.to_numpy().flatten()
        print("SPECT dataset loaded successfully.")
        return X, y
    except Exception as e:
        print(f"Error loading SPECT dataset: {e}")
        return None, None

def load_congress_data():
    """
    Load the Congressional Voting Records dataset from UCI (ID 105).
    
    Returns:
        tuple: (X, y) where X is encoded features and y is labels
    """
    try:
        congress = fetch_ucirepo(id=105)  
        X = congress.data.features
        y = congress.data.targets.to_numpy().flatten()

        X = pd.get_dummies(X).to_numpy()
        print("Congress dataset loaded and encoded successfully.")
        return X, y
    except Exception as e:
        print(f"Error loading Congress dataset: {e}")
        return None, None


results = {}

X_spect, y_spect = load_spect_data()
if X_spect is not None:
    accuracy_spect = process_dataset(X_spect, y_spect, "SPECT")
    results["SPECT"] = accuracy_spect


X_congress, y_congress = load_congress_data()
if X_congress is not None:
    accuracy_congress = process_dataset(X_congress, y_congress, "Congress")
    results["Congress"] = accuracy_congress

print("\nFinal Results:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")