import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.color import rgb2gray
from skimage.transform import resize
import neurokit2 as nk
from .rp_plot import rp_plot

def determine_label(segmentation):
    """
    Helper function to determine the label for a given segmentation.
    """
    if np.any(segmentation == 3):
        return np.array([1, 0, 0])  # AF
    elif np.any(segmentation == 2):
        return np.array([0, 1, 0])  # PRE-AF
    else:
        return np.array([0, 0, 1])  # SR

def gen_rp(data, SEGMENTATION, file, tw, fs, img_dim, delay, dE, rp):
    """
    GEN_RP Generates recurrence plots (RPs) from the input data and saves them to an HDF5 file.

    gen_rp(data, SEGMENTATION, file, tw, fs, img_dim, delay, dE, rp)
    processes the input data to generate recurrence plots using a sliding window approach.
    The generated RPs are saved to the specified HDF5 file along with their corresponding labels.

    Inputs:
        data - The input signal data.
        SEGMENTATION - An array indicating the segmented parts of the input data.
        file - The path to the HDF5 file where RPs and labels will be saved.
        tw - Time window for R-R interval calculation.
        fs - Sampling frequency in Hz.
        img_dim - Dimension of the recurrence plot images.
        delay - Time delay for phase space reconstruction.
        dE - Embedding dimension for phase space reconstruction.
        rp - Boolean flag to display recurrence plot generation figures.

    Outputs:
        None. Results are saved to the specified HDF5 file.

    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
    Early Warning of Atrial Fibrillation Using Deep Learning. 
    Patterns, 2024.
    """
    label_1 = 0
    label_2 = 0
    label_3 = 0
    ini_seg = 0
    N = len(data)
    
    RP_backup = np.zeros(img_dim, dtype=np.int8)
    figure_visibility = 'on' if rp else 'off'


    for ini in range(N // (fs * 15)):
        ini_seg = ini * fs * 15
        end_seg = ini_seg + tw * fs

        if end_seg > N:
            break

        interval = slice(ini_seg, end_seg)
        ecg_w = data[interval]

        try:
            signals, info = nk.ecg_process(ecg_w, sampling_rate=fs,method="pantompkins1985")
            r_peaks = info['ECG_R_Peaks']
            RRI = np.diff(r_peaks) / fs
            RP = rp_plot(RRI, delay, dE)
            rp_min = RP.min()
            rp_max = RP.max()
            RP_norm = ((RP - rp_min) / (rp_max - rp_min) * 255).round().astype(np.uint8)
            RP = RP_norm.astype(np.uint8)

            cmap = plt.get_cmap('jet') 
            RP_colored = cmap(RP_norm)[:, :, :3] # Shape: (N, N, 3)
            RP_gray = rgb2gray(RP_colored)       # Shape: (N, N), Range: 0.0 - 1.0

            scale_factor = 5
            high_res_dim = (img_dim[0] * scale_factor, img_dim[1] * scale_factor)
            RP_high = resize(RP_gray, high_res_dim, order=0, mode='edge', anti_aliasing=False,preserve_range=True)

            RP_final = resize(RP_high, img_dim, order=3, mode='reflect', anti_aliasing=True,preserve_range=True)            # RP = resize(RP, img_dim, order=1, mode='reflect', anti_aliasing=False, preserve_range=True)
            RP = (RP_final * 255).astype(np.uint8)
            RP_backup = RP

        except:

            RP = RP_backup

        with h5py.File(file, 'a') as f:
            x_dataset = f['/x']
            y_dataset = f['/y']

            cur_size = x_dataset.shape[0]   # current number of samples
            x_dataset.resize((cur_size + 1, img_dim[0], img_dim[1]))
            x_dataset[cur_size, :, :] = RP

            label = determine_label(SEGMENTATION[interval])
            y_dataset.resize((cur_size + 1,3))
            y_dataset[cur_size] = label  # or label as float


            if label[0] == 1:
                label_1 += 1
            elif label[1] == 1:
                label_2 += 1
            elif label[2] == 1:
                label_3 += 1

