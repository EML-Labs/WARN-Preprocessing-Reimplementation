import os
import h5py
import numpy as np

from lib_python.data_seg import data_seg
from lib_python.gen_rp import gen_rp

def gen_dat(tw, dim, fs, delay, dE, dis_thresh, inputf, outputf, SAMPLES, rp):
    """
    GEN_DAT Generates datasets and recurrence plots for a list of samples.

    gen_dat(tw, dim, fs, delay, dE, dis_thresh, inputf, outputf, SAMPLES, rp)
    processes a list of sample files, performs data segmentation, and generates
    recurrence plots (RPs) for each sample, saving the results in HDF5 format.

    Inputs:
        tw - Time window for R-R interval calculation.
        dim - Dimensionality of the recurrence plot.
        fs - Sampling frequency in Hz.
        delay - Time delay for phase space reconstruction.
        dE - Embedding dimension for phase space reconstruction.
        dis_thresh - Threshold for the coefficient of variation of R-R intervals.
        inputf - Input file directory.
        outputf - Output file directory.
        SAMPLES - Cell array containing sample identifiers.
        rp - Parameters to display recurrence plot generation.

    Outputs:
        None. Results are saved to HDF5 files.

    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
    Early Warning of Atrial Fibrillation Using Deep Learning. 
    Patterns, 2024.
    """
    for i in range(len(SAMPLES[0])):
        file_read = SAMPLES[0][i] + '.txt'
        output_file_path = os.path.join(outputf, SAMPLES[0][i] + '.hdf5')
        # Check for existence of input and output files
        if not os.path.exists(os.path.join(inputf, file_read)) or os.path.exists(output_file_path):
            print(f"Input file {file_read} does not exist or output file already exists. Skipping...")
            continue

        print(f"Processing file: {file_read}")

        file = os.path.join(outputf, SAMPLES[0][i] + '.hdf5')
        with h5py.File(file, 'w') as f:
            f.create_dataset(
                "/x",
                shape=(0, dim, dim),           # start with 0 samples
                maxshape=(None, dim, dim),     # unlimited along first axis
                dtype='uint8',
                chunks=(1, dim, dim)           # chunk along sample axis
            )

            f.create_dataset(
                "/y",
                shape=(0, 3),                    # start with 0 labels
                maxshape=(None, 3),              # unlimited along first axis
                dtype='float64',
                chunks=(1, 3)
            )


        DATA = np.loadtxt(inputf + file_read)  
        DATA = DATA[:, 1]  

        SEGMENTATION = data_seg(DATA, fs, SAMPLES, dis_thresh, tw, i)
        gen_rp(DATA, SEGMENTATION, file, tw, fs, (dim, dim), delay, dE, rp)
