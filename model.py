import numpy as np


def model_matrix(design):
    """
    Construct the model matrix for a
    second-order response surface model.

    Parameters
    ----------
    design : numpy array
        Experimental points with two columns:
        column 1 = x1 (temperature)
        column 2 = x2 (concentration)

    Returns
    -------
    X : numpy array
        Regression/model matrix.
    """

    design = np.asarray(design, dtype=float)

    # Extract the two experimental factors
    x1 = design[:, 0]
    x2 = design[:, 1]

    # Construct the six model terms
    X = np.column_stack([
        np.ones(len(design)),  # Intercept
        x1,                    # Linear effect of x1
        x2,                    # Linear effect of x2
        x1**2,                 # Quadratic effect of x1
        x2**2,                 # Quadratic effect of x2
        x1*x2                  # Interaction effect
    ])

    return X


def information_matrix(design):
    """
    Calculate the information matrix M = X.T @ X.
    """

    X = model_matrix(design)

    M = X.T @ X

    return M