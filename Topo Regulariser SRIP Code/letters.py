import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the Letters dataset
data = fetch_openml(name='letter', version=1, as_frame=False)
X = data.data  # 16 features
y = data.target  # 26 classes (A-Z)

# Encode labels to integers (0-25)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [5, 10, 20, 30, 40],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

svm = SVC(decision_function_shape='ovo')  # One-vs-one for multi-class classification

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train SVM with best parameters without topological penalty
svm = SVC(**best_params, decision_function_shape='ovo')
svm.fit(X_train, y_train)

# Evaluate accuracy without topological penalty
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)
print(f"Training accuracy without topological penalty: {train_accuracy:.4f}")
print(f"Testing accuracy without topological penalty: {test_accuracy:.4f}")
mean_error_rate = 1.0 - np.mean([train_accuracy, test_accuracy])
print(f"Mean error rate: {mean_error_rate:.4f}")

# Define UnionFind class for topological regularization
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.min_point = [i for i in range(n)]

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.min_point[root_u] = min(self.min_point[root_u], self.min_point[root_v], key=lambda p: sorted_values[p])
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.min_point[root_v] = min(self.min_point[root_u], self.min_point[root_v], key=lambda p: sorted_values[p])
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                self.min_point[root_u] = min(self.min_point[root_u], self.min_point[root_v], key=lambda p: sorted_values[p])

    def find_min(self, u):
        return self.min_point[self.find(u)]

# Compute robustness for a single class (simplified for multi-class)
def compute_robustness(X, svm, class_label):
    values = svm.decision_function(X)[:, class_label]  # Decision function for specific class
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]

    uf = UnionFind(len(X))
    pairings_f = []

    for i in range(1, len(sorted_indices)):
        u = sorted_indices[i]
        v = sorted_indices[i - 1]
        uf.union(u, v)
        min_v = uf.find_min(v)
        if sorted_values[min_v] <= 0 and sorted_values[u] >= 0:
            pairings_f.append((min_v, u))

    neg_values = -values
    neg_sorted_indices = np.argsort(neg_values)
    neg_sorted_values = neg_values[neg_sorted_indices]

    uf_neg = UnionFind(len(X))
    pairings_neg_f = []

    for i in range(1, len(neg_sorted_indices)):
        u = neg_sorted_indices[i]
        v = neg_sorted_indices[i - 1]
        uf_neg.union(u, v)
        min_v = uf_neg.find_min(v)
        if neg_sorted_values[min_v] <= 0 and neg_sorted_values[u] >= 0:
            pairings_neg_f.append((min_v, u))

    critical_pairs = pairings_f + pairings_neg_f
    critical_values = [(values[p1], values[p2]) for p1, p2 in critical_pairs]
    robustness = [min(abs(f_p1), abs(f_p2)) for f_p1, f_p2 in critical_values]
    return robustness

# Compute robustness across all classes
values = svm.decision_function(X_train)
robustness_per_class = []
for class_label in range(26):  # 26 classes
    robustness = compute_robustness(X_train, svm, class_label)
    robustness_per_class.append(robustness)

# Combine robustness (sum of squares across all classes)
robustness = sum(sum(r**2 for r in robustness) for robustness in robustness_per_class)

# Combined loss function
def combined_loss(w, X, y, svm, lambda_):
    svm.set_params(C=w[0])
    svm.fit(X, y)
    decision_function = svm.decision_function(X)
    # Simplified multi-class hinge loss
    correct_class_scores = decision_function[np.arange(len(y)), y]
    hinge_loss = np.sum(np.maximum(0, 1 - correct_class_scores))
    topological_penalty = lambda_ * robustness
    return hinge_loss + topological_penalty

# Gradient of the combined loss function (simplified)
def combined_loss_gradient(w, X, y, svm, lambda_):
    svm.set_params(C=w[0])
    svm.fit(X, y)
    decision_function = svm.decision_function(X)
    correct_class_scores = decision_function[np.arange(len(y)), y]
    hinge_loss_grad = np.sum(-(correct_class_scores < 1).astype(int))
    topological_penalty_grad = 2 * lambda_ * robustness
    return hinge_loss_grad + topological_penalty_grad

# Gradient descent for combined loss function
def gradient_descent(X, y, svm, lambda_, learning_rate=0.01, num_iterations=100):
    w = np.array([1.0])  # Initialize the weight (C parameter)
    epsilon = 0.01  # Small positive value to prevent C from becoming non-positive
    for i in range(num_iterations):
        grad = combined_loss_gradient(w, X, y, svm, lambda_)
        w[0] -= learning_rate * grad
        w[0] = max(w[0], epsilon)  # Ensure C remains positive
    return w

# Perform gradient descent with topological regularization
lambda_ = 0.2
learning_rate = 0.01
num_iterations = 100
optimal_w = gradient_descent(X_train, y_train, svm, lambda_, learning_rate, num_iterations)

# Train SVM with topological penalty
svm.set_params(C=optimal_w[0])
svm.fit(X_train, y_train)

# Evaluate accuracy with topological penalty
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)
print(f"Training accuracy with topological penalty: {train_accuracy:.4f}")
print(f"Testing accuracy with topological penalty: {test_accuracy:.4f}")
mean_error_rate = 1.0 - np.mean([train_accuracy, test_accuracy])
print(f"Mean error rate: {mean_error_rate:.4f}")

# Plotting function (skipped due to high dimensionality)
def plot_svm_boundary(X, y, svm, title):
    print(f"Skipping plot for '{title}' due to high-dimensional data (16 features).")
    # For visualization, consider dimensionality reduction (e.g., PCA) separately.

# Visualize results (plotting skipped)
plot_svm_boundary(X_train, y_train, svm, 'SVM with Topological Regularization')

# Train and visualize SVM without topological penalty for comparison
svm_no_penalty = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo')
svm_no_penalty.fit(X_train, y_train)
plot_svm_boundary(X_train, y_train, svm_no_penalty, 'SVM without Topological Regularization')