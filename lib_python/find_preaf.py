from .get_RRI import get_RRI
import numpy as np

def find_preaf(data, AF_ini, fs, dis_thresh, tw, mean_rri):
    """
    FIND_PREAF Identifies the start index of the pre-atrial fibrillation (pre-AF) phase.

    [pre_af_ini] = find_preaf(data, AF_ini, fs, dis_thresh, tw, mean_rri)
    analyzes a segment of ECG data to determine the beginning of the pre-AF
    phase based on the variability of R-R intervals.

    Inputs:
        data - The ECG data as a vector.
        AF_ini - The start index of the AF event in the data.
        fs - Sampling frequency in Hz.
        dis_thresh - Threshold for the coefficient of variation of R-R intervals.
        tw - Time window for R-R interval calculation.
        mean_rri - The mean R-R interval across the dataset.

    Outputs:
        pre_af_ini - The start index for the pre-AF phase in the data.

    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
    Early Warning of Atrial Fibrillation Using Deep Learning. 
    Patterns, 2024.
    """ 
    m = 60 * fs  # Convert minutes to samples
    pre_af_ini = 0
    flag = True
    count = 1

    interval_start = len(data) - 5 * m
    interval_end = len(data)
    interval = data[interval_start:interval_end]

    while flag and interval_start > 0:
        x = interval
        RRI = get_RRI(x, fs, tw)
        std_RRI = [np.std(rri) for rri in RRI]  # Calculate the standard deviation of R-R intervals

        # Compute the coefficient of variation of R-R intervals
        coef_var = [std / mean_rri for std in std_RRI]

        # Check if the median coefficient of variation is below the threshold
        if np.median(coef_var) <= dis_thresh:
            flag = False
            pre_af_ini = AF_ini - count * 5 * m  # Update the start index for pre-AF
        else:
            # Update the search interval for the next iteration
            interval_end = len(data) - count * 5 * m + 30 * fs
            interval_start = len(data) - (count + 1) * 5 * m
            interval = data[interval_start:interval_end]
            count += 1

    return pre_af_ini