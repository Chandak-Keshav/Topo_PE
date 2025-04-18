import numpy as np

def build_weighted_complex(image):
    # Get image dimensions
    height, width = image.shape
    
    # Find nonzero locations in the image
    row, col = np.nonzero(image)
    
    # Define vertex locations for the centers of the pixels
    # Adjust coordinates to match MATLAB's Euclidean system (y increases upwards)
    # MATLAB: col (1-based), height - row; Python: col (0-based), adjust accordingly
    V_centers = np.column_stack((col + 1.0, (height - 1) - row))
    num_centers = V_centers.shape[0]
    
    # Assign weights to center vertices based on grayscale values
    # In MATLAB, image(height - V_centers(j,2), V_centers(j,1)) accesses the original pixel
    # Since V_centers(j,2) = height - row, height - V_centers(j,2) = row (1-based in MATLAB)
    # In Python, row and col are 0-based, so use them directly
    V_center_weights = image[row, col]
    
    # Define corner vertices for each center vertex
    V_corners = []
    for j in range(num_centers):
        center_vertex = V_centers[j]
        V_corners.append([center_vertex[0] - 0.5, center_vertex[1] - 0.5])  # SW
        V_corners.append([center_vertex[0] + 0.5, center_vertex[1] - 0.5])  # SE
        V_corners.append([center_vertex[0] - 0.5, center_vertex[1] + 0.5])  # NW
        V_corners.append([center_vertex[0] + 0.5, center_vertex[1] + 0.5])  # NE
    V_corners = np.array(V_corners)
    
    # Remove duplicate corner vertices
    V_corners = np.unique(V_corners, axis=0)
    
    # Combine center and corner vertices
    V = np.vstack((V_centers, V_corners))
    
    # Define faces (four triangles per center vertex)
    F = []
    for vertex_ind in range(num_centers):
        vertex = V[vertex_ind]
        # Find indices of the four corner neighbors
        NE_neighbor = vertex + [0.5, 0.5]
        NE_neighbor_index = np.where(np.all(V == NE_neighbor, axis=1))[0][0]
        NW_neighbor = vertex + [-0.5, 0.5]
        NW_neighbor_index = np.where(np.all(V == NW_neighbor, axis=1))[0][0]
        SE_neighbor = vertex + [0.5, -0.5]
        SE_neighbor_index = np.where(np.all(V == SE_neighbor, axis=1))[0][0]
        SW_neighbor = vertex + [-0.5, -0.5]
        SW_neighbor_index = np.where(np.all(V == SW_neighbor, axis=1))[0][0]
        # Add four triangular faces
        F.append([vertex_ind, NE_neighbor_index, NW_neighbor_index])
        F.append([vertex_ind, NW_neighbor_index, SW_neighbor_index])
        F.append([vertex_ind, SW_neighbor_index, SE_neighbor_index])
        F.append([vertex_ind, SE_neighbor_index, NE_neighbor_index])
    F = np.array(F)
    
    # Extract unique edges from the triangulation
    all_edges = []
    for tri in F:
        all_edges.append([tri[0], tri[1]])
        all_edges.append([tri[1], tri[2]])
        all_edges.append([tri[2], tri[0]])
    all_edges = np.array(all_edges)
    E = np.unique(np.sort(all_edges, axis=1), axis=0)
    
    # Define face weights (each center vertex contributes to four faces)
    F_weights = np.repeat(V_center_weights, 4)
    
    # Define vertex weights
    # Center vertices keep their weights; corner vertices get max weight of containing faces
    V_weights = V_center_weights.copy()
    for vertex_ind in range(num_centers, len(V)):
        # Find faces containing this vertex
        row = np.where(np.any(F == vertex_ind, axis=1))[0]
        V_weights = np.append(V_weights, np.max(F_weights[row]))
    
    # Define edge weights
    # Each edge weight is the max face weight of faces containing the edge
    E_weights = []
    for edge in E:
        # Find faces where both edge vertices appear (sum = 2)
        face_inds = np.where(np.sum(np.isin(F, edge), axis=1) == 2)[0]
        E_weights.append(np.max(F_weights[face_inds]))
    E_weights = np.array(E_weights)
    
    # Return the simplicial complex components
    return V, E, F, V_weights, E_weights, F_weights