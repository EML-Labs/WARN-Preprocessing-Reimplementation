# EARLY WARNING OF ATRIAL FIBRILLATION USING DEEP LEARNING
# 
#    This script generates the data for training and testing a deep learning
#    model aimed at early detection of atrial fibrillation from ECG data.
#    It processes ECG signals to generate recurrence plots (RPs) and labels
#    them based on the presence of atrial fibrillation.
# 
#    Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., ... & Goncalves, J. 
#    Early Warning of Atrial Fibrillation Using Deep Learning. 
#    Patterns, 2024.

import os,csv
from lib_python.gen_dat import gen_dat
# Parameters
# Configuration parameters for data generation
rp = 0;          # Flag to display recurrence plots: 0=No, 1=Yes
tw = 30;         # Time window size in seconds for RP generation
dim = 224;       # Dimension of the square images for the RPs
fs = 128;        # Sampling frequency of the ECG data in Hz
delay = 2;       # Time delay for phase space reconstruction
dE = 3;          # Embedding dimension for phase space reconstruction
dis_tresh = 0.7; # Threshold for the coefficient of variation for segmentation

# Directories
# Specify input and output directories
inputf = "data/";    # Directory containing the raw ECG data and labels
outputf = "rp_data/"; # Output directory for the generated RPs
os.mkdir(outputf) if not os.path.exists(outputf) else None  # Create the output directory if it does not exist

# Load Samples
# Load the labels for the ECG samples from a CSV file
Samples = []
with open(os.path.join(inputf, "LABELS.csv"), 'r') as fileID:
    reader = csv.DictReader(fileID)
    id = []
    af1 = []
    af2 = []
    af3 = []
    af4 = []
    af5 = []
    af6 = []
    for row in reader:
        id.append(row['ID'])
        af1.append(list(map(int,row['AF1'].replace('[','').replace(']','').split())))
        af2.append(list(map(int,row['AF2'].replace('[','').replace(']','').split())))
        af3.append(list(map(int,row['AF3'].replace('[','').replace(']','').split())))
        af4.append(list(map(int,row['AF4'].replace('[','').replace(']','').split())))
        af5.append(list(map(int,row['AF5'].replace('[','').replace(']','').split())))
        af6.append(list(map(int,row['AF6'].replace('[','').replace(']','').split())))
    Samples = [id, af1, af2, af3, af4, af5, af6]

gen_dat(tw, dim, fs, delay, dE, dis_tresh, inputf, outputf, Samples, rp)