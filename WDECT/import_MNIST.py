import pandas as pd
import numpy as np

def import_MNIST(file, digit):
    M = pd.read_csv(file)
    M = M.to_numpy()
    M = M[M[:, 0] == digit]
    M = M[:100, 1:785]  # Shape: (100, 784)
    data = np.array([row.reshape(28, 28) for row in M])  # Shape: (100, 28, 28)
    data = np.transpose(data, (1, 2, 0))  # Shape: (28, 28, 100)
    return data