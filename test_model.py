import numpy as np

from src.model import model_matrix, information_matrix


# Define five experimental points
design = np.array([
    [-1, -1],
    [-1,  1],
    [ 0,  0],
    [ 1, -1],
    [ 1,  1]
])


# Generate the model matrix
X = model_matrix(design)

print("Model Matrix:")
print(X)

print("\nShape of X:")
print(X.shape)


# Calculate information matrix
M = information_matrix(design)

print("\nInformation Matrix:")
print(M)

print("\nShape of M:")
print(M.shape)