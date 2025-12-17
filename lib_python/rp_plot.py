import numpy as np
def rp_plot(data, delay, dE):
    """
    RP_PLOT Generates a recurrence plot from R-R interval data.

    RP = rp_plot(data, delay, dE) computes the recurrence plot (RP) for the given
    R-R interval time series data using phase space reconstruction with specified
    time delay and embedding dimension.

    Inputs:
        data - 1D array of R-R intervals.
        delay - Time delay for phase space reconstruction.
        dE - Embedding dimension for phase space reconstruction.
    Outputs:
        RP - 2D array representing the recurrence plot.

    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J.
    Early Warning of Atrial Fibrillation Using Deep Learning.
    Patterns, 2024.
    """

    N = len(data)
    Nrp = N - (dE - 1) * delay  # RP size

    # Phase space reconstruction using embedding
    Xdim = np.zeros((Nrp, dE))
    for dim in range(dE):
        Xdim[:, dim] = data[dim * delay : dim * delay + Nrp]

    # Compute the recurrence matrix
    RP = np.zeros((Nrp, Nrp))
    for i in range(Nrp):
        for j in range(Nrp):
            # Sum of squared differences across all dimensions
            RP[i, j] = np.sum((Xdim[i, :] - Xdim[j, :]) ** 2)

    # Convert distances to similarities (optional)
    RP = np.sqrt(RP)  # Euclidean distance
    # For binary recurrence plot, apply a threshold
    # RP = RP <= threshold;

    return RP


