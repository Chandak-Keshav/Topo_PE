import numpy as np

def distance_RotationInvariant(WECT1, WECT2):
    """
    Compute the rotation-invariant distance between two WECTs or SWECTs.

    Parameters:
    - WECT1: First WECT matrix (num_steps x num_directions)
    - WECT2: Second WECT matrix (num_steps x num_directions)

    Returns:
    - dist: Minimal L2 (Frobenius) distance after cyclic shifts
    - shift: The best shift (1 to num_directions) that achieves the minimal distance
    """
    # Ensure WECT1 and WECT2 have the same shape
    assert WECT1.shape == WECT2.shape, "WECT1 and WECT2 must have the same shape"
    
    # Get the number of directions (number of columns)
    num_directions = WECT1.shape[1]
    
    # Initialize an array to store distances for each shift
    distances = np.zeros(num_directions)
    
    # Loop over all possible shifts (1 to num_directions)
    for d in range(1, num_directions + 1):
        # Cyclically shift WECT2 to the left by d positions along the columns (axis=1)
        WECT2_shifted = np.roll(WECT2, -d, axis=1)
        # Compute the Frobenius norm of the difference and store it
        distances[d-1] = np.linalg.norm(WECT1 - WECT2_shifted, 'fro')
    
    # Find the minimal distance
    dist = np.min(distances)
    # Find the shift (convert from 0-based to 1-based indexing)
    shift = np.argmin(distances) + 1
    
    return dist, shift