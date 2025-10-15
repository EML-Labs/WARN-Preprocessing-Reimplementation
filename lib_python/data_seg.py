import numpy as np
from lib_python.get_RRI import get_RRI
from lib_python.find_preaf import find_preaf
from scipy.stats import trim_mean

def data_seg(DATA, fs, LABELS, dis_thresh, tw, n_file):
    """
    DATA_SEG Segments the input data based on atrial fibrillation (AF) events.

    [SEGMENTATION] = data_seg(DATA, fs, LABELS, dis_thresh, tw, n_file)
    segments the input DATA into sinus rhythm (SR), pre-AF, and AF phases
    based on LABELS provided for atrial fibrillation events.

    Inputs:
        DATA - The input signal data.
        fs - Sampling frequency in Hz.
        LABELS - Cell array containing AF event labels and times.
        dis_thresh - Threshold for the coefficient of variation of R-R intervals.
        tw - Time window for R-R interval calculation.
        n_file - File index to select the correct labels.

    Outputs:
        SEGMENTATION - An array indicating the segmented parts of the input data.
                          1 = Sinus Rhythm (SR), 2 = Pre-AF, 3 = AF


    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
    Early Warning of Atrial Fibrillation Using Deep Learning. 
    Patterns, 2024."""

    N = len(DATA)
    m = fs * 60 # Sampling frequency in minutes
    SR_ini = 0 # Start index for sinus rhythm
    SEGMENTATION = np.ones(N) # Initialize segmentation array with SR (1)

    RR_T = get_RRI(DATA, fs, tw) # Get R-R intervals
    mean_rri = trim_mean(np.hstack(RR_T), 0.05) # Trimmed mean of R-R intervals

    for i in range(1, len(LABELS)):
        if len(LABELS[i][n_file]) > 0:
            AF_ini = LABELS[i][n_file][0]
            AF_end = LABELS[i][n_file][1]

            # Detect Pre-AF Segment
            PreAF = find_preaf(DATA[SR_ini:AF_ini], AF_ini, fs, dis_thresh, tw, mean_rri)

            SEGMENTATION[SR_ini:AF_ini] = 1 # SR segment
            SEGMENTATION[PreAF+1:AF_ini] = 2 # Pre-AF segment
            SEGMENTATION[AF_ini:AF_end] = 3 # AF segment

            SR_ini = AF_end + 1
        else:
            SR_end = N
            SEGMENTATION[SR_ini:SR_end] = 1 # SR segment
            break

    return SEGMENTATION

