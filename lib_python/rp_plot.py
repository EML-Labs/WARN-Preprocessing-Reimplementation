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

    # Basic input validation
    if data is None:
        raise ValueError("`data` must not be None.")

    # Convert to NumPy array for consistent handling
    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("`data` must not be empty.")

    if not isinstance(delay, int) or delay <= 0:
        raise ValueError("`delay` must be a positive integer.")

    if not isinstance(dE, int) or dE <= 0:
        raise ValueError("`dE` (embedding dimension) must be a positive integer.")

    N = data.size
    Nrp = N - (dE - 1) * delay  # RP size
    if Nrp <= 0:
        raise ValueError(
            "Length of `data` is insufficient for the given `delay` and `dE`. "
            f"Got len(data)={N}, delay={delay}, dE={dE}."
        )

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


