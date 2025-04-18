import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from import_MNIST import import_MNIST
from build_weighted_complex import build_weighted_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT, weighted_euler_curve
from distance_RotationInvariant import distance_RotationInvariant

# Load MNIST Data with Explicit Labels
dataset = {}
labels = []  # Store true labels explicitly
for j in range(10):  # Digits 0-9
    MNISTdata = import_MNIST('mnist_test.csv', j)
    for k in range(100):  # 100 images per digit
        dataset[(j, k)] = MNISTdata[:, :, k].T
        labels.append(j)  # Label corresponds to digit j
    print(f'Digit {j} loaded...')

# Convert a Sample Image to Weighted Simplicial Complex (for verification)
image = dataset[(2, 4)]
V, E, F, V_weights, E_weights, F_weights = build_weighted_complex(image)
complex = {
    'V': V, 'E': E, 'F': F,
    'V_weights': V_weights, 'E_weights': E_weights, 'F_weights': F_weights
}

# Plot Sample Image and Complex
plt.figure(1)
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image (Digit 2)')
plt.axis('equal')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.scatter(V[:, 0], V[:, 1], c=V_weights, cmap='viridis')
plt.title('Weighted Simplicial Complex')
plt.axis('equal')
plt.axis('off')
plt.show()

# Compute Smoothed WECT for Sample
num_directions = 25
num_steps = 50
method = 'gaussian'
window = 0.2 * num_steps
normalization_method = 'none'
SWECT = complex_to_weighted_ECT(complex, num_directions, num_steps, method, window, normalization_method)

# Plot Smoothed Weighted Euler Curves
plt.figure(2)
plt.clf()
plt.plot(SWECT)
plt.title('Smoothed Weighted Euler Curves (Digit 2)')
plt.show()

# Build Complexes for All MNIST Digits
all_MNIST_complexes = []
for j in range(10):
    for k in range(100):
        image = dataset[(j, k)]
        V, E, F, V_weights, E_weights, F_weights = build_weighted_complex(image)
        complex = {
            'V': V, 'E': E, 'F': F,
            'V_weights': V_weights, 'E_weights': E_weights, 'F_weights': F_weights
        }
        all_MNIST_complexes.append(complex)
    print(f'Digit {j} complexes built...')

# Compute Smoothed WECTs for All Complexes
SWECTs = []
for n in range(1000):  # 10 digits * 100 images
    complex = all_MNIST_complexes[n]
    SWECT = complex_to_weighted_ECT(complex, num_directions, num_steps, method, window, normalization_method)
    SWECTs.append(SWECT)
    if n % 100 == 0:
        print(f'WECT {n} computed...')

# Compute Pairwise Distances Modulo Rotation
distMat_modRotations = np.zeros((1000, 1000))
for j in range(1000):
    for k in range(j + 1, 1000):
        dist, _ = distance_RotationInvariant(SWECTs[j], SWECTs[k])
        distMat_modRotations[j, k] = dist
distMat_modRotations = distMat_modRotations + distMat_modRotations.T  # Symmetric matrix

# Visualize Distance Matrix
plt.figure(4)
plt.clf()
plt.imshow(distMat_modRotations, cmap='viridis')
plt.title('Distances Modulo Rotation Between Digits')
plt.colorbar()
plt.show()

# kNN Classification
K = 10  # Number of neighbors
matrix = distMat_modRotations

# Find indices of K nearest neighbors (excluding self)
inds = np.argsort(matrix, axis=1)
neighbor_inds = inds[:, 1:K+1]  # Skip index 0 (self)

# Convert neighbor indices to classes using explicit labels
labels = np.array(labels)  # Convert to NumPy array for efficiency
classification = np.zeros(1000, dtype=int)
for j in range(1000):
    neighbor_classes = labels[neighbor_inds[j]]  # Get classes of neighbors
    classification[j] = np.bincount(neighbor_classes).argmax()  # Most frequent class

# Compute Classification Rates per Digit
class_rates = np.zeros(10)
for digit in range(10):
    true_labels = labels[digit * 100:(digit + 1) * 100]  # True labels for this digit
    pred_labels = classification[digit * 100:(digit + 1) * 100]  # Predicted labels
    class_rates[digit] = np.mean(pred_labels == true_labels)  # Accuracy for this digit

# Overall Classification Rate
overall_classification_rate = np.mean(class_rates)
print(f'Overall Classification Rate: {overall_classification_rate:.4f}')

# Compute and Plot Confusion Matrix
confusion = np.zeros((10, 10))
for j in range(10):  # True digit
    for k in range(10):  # Predicted digit
        confusion[j, k] = np.sum(classification[j * 100:(j + 1) * 100] == k) / 100
