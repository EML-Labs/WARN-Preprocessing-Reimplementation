import neurokit2 as nk
import numpy as np

def get_RRI(data, fs, tw):
    """
    GET_RRI Calculates R-R intervals (RRI) from ECG data.

    [RRI_ALL] = get_RRI(data, fs, tw) calculates R-R intervals over specified
    time windows from ECG data using the Pan-Tompkins algorithm.

    Inputs:
        data - The ECG data as a vector.
        fs - Sampling frequency in Hz.
        tw - Time window for processing in seconds.

    Outputs:
        RRI_ALL - Cell array of R-R intervals calculated over each time window.

    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
    Early Warning of Atrial Fibrillation Using Deep Learning. 
    Patterns, 2024."""
    pass

    RRI_ALL = []  # List to store R-R intervals from each window
    data_len = len(data)  # Length of the ECG data
    ini = 0  # Initial index for storing RRI

    while True:
        end_idx = data_len - (ini * fs * 5)
        start_idx = max(0, end_idx - (tw * fs))

        if start_idx == 0:
            break

        ecg_w = data[start_idx:end_idx]  # Extract the window of ECG data

        # Use NeuroKit2 to find R-peaks using the Pan-Tompkins algorithm
        signals, info = nk.ecg_process(ecg_w, sampling_rate=fs)
        r_peaks = info['ECG_R_Peaks']

        # Calculate R-R intervals (RRI) for the current window
        RRI = np.diff(r_peaks) / fs

        # Store the calculated RRI
        RRI_ALL.append(RRI)

        ini += 1  # Update the index for the next window

    RRI_ALL = RRI_ALL[::-1]  # Reverse RRI_ALL to have RRI in chronological order
    return RRI_ALL
