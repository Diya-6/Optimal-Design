
import numpy as np

from src.model import model_matrix, information_matrix


def check_information_matrix(M):
    """
    Check whether the information matrix is
    positive definite and numerically usable.
    """

    # Information matrix must be symmetric
    if not np.allclose(M, M.T):
        raise ValueError("Information matrix is not symmetric.")

    # Calculate eigenvalues of the symmetric matrix
    eigenvalues = np.linalg.eigvalsh(M)

    # Reject singular or numerically near-singular matrices
    tolerance = 1e-10 * max(1.0, np.max(np.abs(eigenvalues)))

    if eigenvalues[0] <= tolerance:
        raise ValueError(
            "Design is singular or numerically unstable. "
            "All model parameters cannot be reliably estimated."
        )

    return eigenvalues


def d_optimality(M):
    """
    D-optimality:
    Maximize log(det(M)).
    """

    sign, log_det = np.linalg.slogdet(M)

    if sign <= 0:
        raise ValueError("Information matrix must have positive determinant.")

    return log_det


def a_optimality(M):
    """
    A-optimality:
    Minimize trace(inv(M)).
    """

    # Solve MZ = I instead of explicitly calculating inv(M)
    identity = np.eye(M.shape[0])

    M_inv = np.linalg.solve(M, identity)

    return np.trace(M_inv)


def e_optimality(M):
    """
    E-optimality:
    Maximize the minimum eigenvalue of M.
    """

    eigenvalues = np.linalg.eigvalsh(M)

    return eigenvalues[0]


def i_optimality(M, prediction_grid):
    """
    I-optimality:
    Minimize average prediction variance
    over the experimental region.
    """

    # Construct the model matrix for prediction points
    F = model_matrix(prediction_grid)

    # Solve MZ = F.T
    Z = np.linalg.solve(M, F.T)

    # Calculate f(x).T @ inv(M) @ f(x)
    prediction_variances = np.sum(F * Z.T, axis=1)

    # Average over all prediction points
    return np.mean(prediction_variances)


def calculate_all_criteria(design, prediction_grid):
    """
    Calculate all four optimality criteria
    for a given experimental design.

    Returns
    -------
    results : dictionary
        D, A, E and I criterion values.
    """

    # Construct information matrix
    M = information_matrix(design)

    # Verify that the design is estimable
    eigenvalues = check_information_matrix(M)

    # Calculate each criterion
    results = {
        "D": d_optimality(M),
        "A": a_optimality(M),
        "E": e_optimality(M),
        "I": i_optimality(M, prediction_grid)
    }

    return results
