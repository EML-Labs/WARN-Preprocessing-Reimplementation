import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def pan_tom(ecg, fs, gr=0):
    pass
# def pan_tom(ecg, fs, gr=0):
#     """
#     Python implementation of Pan-Tompkins ECG QRS detector.
#     Returns:
#       qrs_amp_raw : amplitudes of detected R peaks (from bandpass signal)
#       qrs_i_raw   : indices (sample locations) of detected R peaks
#       delay       : accumulated algorithm delay (in samples)
#     """
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)

#     ecg = np.asarray(ecg).flatten()
#     if ecg.ndim != 1:
#         raise ValueError("ecg must be a 1D array")

#     delay = 0
#     skip = 0
#     m_selected_RR = 0
#     mean_RR = 0
#     ser_back = 0

#     # --- Filtering stage (noise cancellation) ---
#     if fs == 200:
#         ecg = ecg - np.mean(ecg)
#         ecg_l, d1 = low_passfilter(ecg, fs)
#         ecg_h, d2 = high_passfilter(ecg_l, fs)
#         ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-12)
#         delay += d1 + d2
#         # plotting placeholders handled later
#     else:
#         ecg_h = bandpass_filter(ecg, fs, f1=5, f2=15)
#         ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-12)

#     # --- Derivative ---
#     ecg_d = derivative_filter(ecg_h, fs)

#     # --- Squaring ---
#     ecg_s = squaring(ecg_d)

#     # --- Moving window integration ---
#     ecg_m, d3 = moving_window_integration(ecg_s, fs)
#     delay += d3

#     # --- Peak detection (on the MWI signal) ---
#     min_peak_dist = int(round(0.2 * fs))  # 200 ms / minimum distance between peaks
#     # Use scipy find_peaks to locate candidate peaks in ecg_m
#     pks_idxs, _ = find_peaks(ecg_m, distance=min_peak_dist)
#     pks = ecg_m[pks_idxs]
#     locs = pks_idxs

#     LLp = len(pks)
#     if LLp == 0:
#         return np.array([]), np.array([]), delay

#     # initialize arrays as in MATLAB
#     qrs_c = np.zeros(LLp)
#     qrs_i = np.zeros(LLp, dtype=int)
#     qrs_i_raw = np.zeros(LLp, dtype=int)
#     qrs_amp_raw = np.zeros(LLp)

#     nois_c = np.zeros(LLp)
#     nois_i = np.zeros(LLp)

#     SIGL_buf = np.zeros(LLp)
#     NOISL_buf = np.zeros(LLp)
#     SIGL_buf1 = np.zeros(LLp)
#     NOISL_buf1 = np.zeros(LLp)
#     THRS_buf1 = np.zeros(LLp)
#     THRS_buf  = np.zeros(LLp)

#     # training phase (first 2 seconds)
#     init_len = min(len(ecg_m), 2 * fs)
#     THR_SIG = np.max(ecg_m[:init_len]) * (1.0 / 3.0)
#     THR_NOISE = np.mean(ecg_m[:init_len]) * (1.0 / 2.0)
#     SIG_LEV = THR_SIG
#     NOISE_LEV = THR_NOISE

#     THR_SIG1 = np.max(ecg_h[:init_len]) * (1.0 / 3.0)
#     THR_NOISE1 = np.mean(ecg_h[:init_len]) * (1.0 / 2.0)
#     SIG_LEV1 = THR_SIG1
#     NOISE_LEV1 = THR_NOISE1

#     Beat_C = 0
#     Beat_C1 = 0
#     Noise_Count = 0

#     # loop through candidate peaks
#     for i in range(LLp):
#         loc = locs[i]
#         # locate corresponding peak in bandpassed signal ecg_h in window [loc - 150ms, loc]
#         win = int(round(0.150 * fs))
#         start = max(loc - win, 0)
#         end = min(loc, len(ecg_h) - 1)
#         if start <= end:
#             # find local max in ecg_h window
#             sub = ecg_h[start:end + 1]
#             if len(sub) == 0:
#                 y_i = 0
#                 x_i = 0
#             else:
#                 x_local = np.argmax(sub)
#                 y_i = sub[x_local]
#                 x_i = x_local + 1  # MATLAB style 1-based local
#         else:
#             y_i = 0
#             x_i = 0

#         # update heart rate logic
#         if Beat_C >= 9:
#             diffRR = np.diff(qrs_i[Beat_C - 8:Beat_C].astype(float))
#             mean_RR = np.mean(diffRR) if len(diffRR) > 0 else mean_RR
#             comp = qrs_i[Beat_C - 1] - qrs_i[Beat_C - 2]
#             if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
#                 THR_SIG = 0.5 * THR_SIG
#                 THR_SIG1 = 0.5 * THR_SIG1
#             else:
#                 m_selected_RR = mean_RR

#         # search-back if long RR detected
#         test_m = m_selected_RR if m_selected_RR else (mean_RR if mean_RR else 0)
#         if test_m:
#             if Beat_C > 0 and (loc - qrs_i[Beat_C - 1]) >= round(1.66 * test_m):
#                 # search back between the last detected QRS and current loc (excluding 200ms windows)
#                 search_start = int(qrs_i[Beat_C - 1] + round(0.200 * fs))
#                 search_end = int(loc - round(0.200 * fs))
#                 if search_start < search_end and search_start < len(ecg_m):
#                     pks_seg = ecg_m[search_start: min(search_end + 1, len(ecg_m))]
#                     if len(pks_seg) > 0:
#                         pks_temp = np.max(pks_seg)
#                         locs_temp = search_start + np.argmax(pks_seg)
#                         if pks_temp > THR_NOISE:
#                             # treat as beat
#                             qrs_c[Beat_C] = pks_temp
#                             qrs_i[Beat_C] = locs_temp
#                             # locate in ecg_h
#                             s = max(locs_temp - win, 0)
#                             e = min(locs_temp, len(ecg_h) - 1)
#                             sub2 = ecg_h[s:e + 1]
#                             if len(sub2) > 0:
#                                 x_i_t = np.argmax(sub2)
#                                 y_i_t = sub2[x_i_t]
#                                 qrs_i_raw[Beat_C] = s + x_i_t
#                                 qrs_amp_raw[Beat_C] = y_i_t
#                                 SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1
#                             SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV

#         # classify peak as signal or noise
#         if pks[i] >= THR_SIG:
#             # check for T wave (if within 360 ms of previous QRS)
#             if Beat_C >= 3 and (loc - qrs_i[Beat_C - 1]) <= round(0.3600 * fs):
#                 slope1_start = max(loc - int(round(0.075 * fs)), 0)
#                 slope1_end = loc
#                 slope2_start = max(int(qrs_i[Beat_C - 1] - round(0.075 * fs)), 0)
#                 slope2_end = int(qrs_i[Beat_C - 1])
#                 slope1 = np.mean(np.diff(ecg_m[slope1_start:slope1_end + 1])) if (slope1_end - slope1_start) > 0 else 0
#                 slope2 = np.mean(np.diff(ecg_m[slope2_start:slope2_end + 1])) if (slope2_end - slope2_start) > 0 else 0
#                 if abs(slope1) <= abs(0.5 * slope2):
#                     # T wave -> noise
#                     Noise_Count += 1
#                     if Noise_Count <= LLp:
#                         nois_c[Noise_Count - 1] = pks[i]
#                         nois_i[Noise_Count - 1] = loc
#                     skip = 1
#                     NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
#                     NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
#                 else:
#                     skip = 0

#             if skip == 0:
#                 # count as QRS in MWI signal
#                 qrs_c[Beat_C] = pks[i]
#                 qrs_i[Beat_C] = loc
#                 # bandpass check
#                 if y_i >= THR_SIG1:
#                     qrs_i_raw[Beat_C1] = (loc - win + (x_i - 1)) if (loc - win + (x_i - 1)) >= 0 else loc
#                     qrs_amp_raw[Beat_C1] = y_i
#                     SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1
#                     Beat_C1 += 1
#                 SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV
#                 Beat_C += 1
#         elif (THR_NOISE <= pks[i]) and (pks[i] < THR_SIG):
#             NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
#             NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
#         else:
#             # pks < THR_NOISE
#             Noise_Count += 1
#             if Noise_Count <= LLp:
#                 nois_c[Noise_Count - 1] = pks[i]
#                 nois_i[Noise_Count - 1] = loc
#             NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
#             NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

#         # update thresholds
#         if NOISE_LEV != 0 or SIG_LEV != 0:
#             THR_SIG = NOISE_LEV + 0.25 * abs(SIG_LEV - NOISE_LEV)
#             THR_NOISE = 0.5 * THR_SIG
#         if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
#             THR_SIG1 = NOISE_LEV1 + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
#             THR_NOISE1 = 0.5 * THR_SIG1

#         SIGL_buf[i] = SIG_LEV
#         NOISL_buf[i] = NOISE_LEV
#         THRS_buf[i] = THR_SIG
#         SIGL_buf1[i] = SIG_LEV1
#         NOISL_buf1[i] = NOISE_LEV1
#         THRS_buf1[i] = THR_SIG1

#         # reset
#         skip = 0
#         ser_back = 0

#     # trim results to detected counts (Beat_C1 and Beat_C)
#     qrs_i_raw = qrs_i_raw[:Beat_C1].astype(int)
#     qrs_amp_raw = qrs_amp_raw[:Beat_C1]
#     qrs_c = qrs_c[:Beat_C]
#     qrs_i = qrs_i[:Beat_C].astype(int)
#     return qrs_amp_raw, qrs_i_raw, delay

def low_passfilter(ecg, fs):
    """
    Low-pass (or part of bandpass) as used in the MATLAB version.
    For fs == 200, uses a lowpass at 12 Hz (order 3) with zero-phase filtering.
    Otherwise this function is not used (bandpass used instead).
    Returns filtered signal and approximate delay contributed by this stage.
    """
    if fs == 200:
        Wn = 12 * 2.0 / fs
        N = 3
        b, a = butter(N, Wn, btype='low')
        ecg_l = filtfilt(b, a, ecg)
        ecg_l = ecg_l / np.max(np.abs(ecg_l)) if np.max(np.abs(ecg_l)) != 0 else ecg_l
        delay = 0  # filtfilt is zero-phase so practical delay is handled later
        return ecg_l, delay
    else:
        # not used for other fs in original code path
        return ecg, 0

def high_passfilter(ecg, fs):
    """
    High-pass (or bandpass complement) used for fs==200 path.
    For fs != 200, a bandpass filter is applied outside.
    Returns filtered signal and approximate delay contributed by this stage.
    """
    if fs == 200:
        Wn = 5 * 2.0 / fs
        N = 3
        b, a = butter(N, Wn, btype='high')
        ecg_h = filtfilt(b, a, ecg)
        ecg_h = ecg_h / np.max(np.abs(ecg_h)) if np.max(np.abs(ecg_h)) != 0 else ecg_h
        return ecg_h, 0
    else:
        return ecg, 0
    
def bandpass_filter(ecg, fs, f1=5, f2=15):
    """
    Bandpass used when fs != 200 (and also a direct alternative).
    3rd-order Butterworth.
    """
    Wn = [f1 * 2.0 / fs, f2 * 2.0 / fs]
    N = 3
    b, a = butter(N, Wn, btype='band')
    ecg_bp = filtfilt(b, a, ecg)
    ecg_bp = ecg_bp / np.max(np.abs(ecg_bp)) if np.max(np.abs(ecg_bp)) != 0 else ecg_bp
    return ecg_bp

def derivative_filter(ecg_h, fs):
    """
    Derivative filter approximating Pan-Tompkins derivative:
    kernel ~ [1 2 0 -2 -1] * (1/8T) where T = 1/fs.
    Implementation follows MATLAB: if fs==200 use fixed kernel; otherwise interpolate.
    """
    if fs != 200:
        # emulate the MATLAB approach: create an interpolated kernel
        int_c = (5 - 1) / (fs * 1.0 / 40.0)  # same formula as MATLAB's int_c
        base = np.array([1, 2, 0, -2, -1]) * (1.0 / 8.0) * fs
        # create indices 1..5 and interpolate at step 1:int_c:5
        idx_src = np.arange(1, 6)
        idx_target = np.arange(1, 5 + 1e-12, int_c)
        b = np.interp(idx_target, idx_src, base)
    else:
        b = np.array([1, 2, 0, -2, -1]) * (1.0 / 8.0) * fs

    # use filtfilt for zero-phase (MATLAB used filtfilt)
    ecg_d = filtfilt(b, [1.0], ecg_h)
    if np.max(np.abs(ecg_d)) != 0:
        ecg_d = ecg_d / np.max(np.abs(ecg_d))
    return ecg_d

def squaring(ecg_d):
    """
    Squaring as per Pan-Tompkins algorithm.
    """
    ecg_s = ecg_d ** 2
    return ecg_s



def moving_window_integration(ecg_s, fs):
    N = int(round(0.150 * fs))  # 150 ms window
    if N < 1:
        N = 1
    window = np.ones(N) / N
    ecg_m = np.convolve(ecg_s, window, mode='full')
    # keep same length as input by truncation (MATLAB conv gave longer and indices referenced accordingly)
    ecg_m = ecg_m[:len(ecg_s)]
    delay = N // 2
    return ecg_m, delay


def find_peaks(ecg, fs):
    """
    Peak detection as per Pan-Tompkins algorithm.
    """
    pass