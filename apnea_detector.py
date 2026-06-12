"""
apnea_detector.py

Reusable functions for loading, resampling (to 250 Hz), filtering, and detecting apnea intervals
from mask pressure CSV files (Inline_PQ_Data).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def load_and_resample_mask(file_path, target_fs=250.0):
    """
    Load a mask pressure CSV, sanitize the data, and resample both 
    pressure channels to a uniform time base.
    """
    data = pd.read_csv(file_path)
        # 1. Drop any rows where the time or pressure data is missing
    data = data.dropna(subset=["Time [s]", "Gauge Pressure [cmH2O]", "Inspiratory differential pressure [cmH2O]"])
    
    # 2. Sort by time to ensure it is strictly increasing
    data = data.sort_values("Time [s]")
    
    # 3. Drop duplicate timestamps (prevents SciPy divide-by-zero interpolation error)
    data = data.drop_duplicates(subset=["Time [s]"])
    
    time_raw = data["Time [s]"].values
    
    # 4. Guard against empty files or files with 0-second duration 
    # (prevents the "index -1 out of bounds" error)
    if len(time_raw) < 2 or (time_raw[-1] <= time_raw[0]):
        raise ValueError("Corrupted file: Insufficient data points or zero-duration time span.")

    gauge_raw = data["Gauge Pressure [cmH2O]"].values
    insp_raw = data["Inspiratory differential pressure [cmH2O]"].values

    dt = 1.0 / target_fs
    t_new = np.arange(time_raw[0], time_raw[-1], dt)

    if len(t_new) == 0:
         raise ValueError("Resampled time array is empty.")

    interp_gauge = interp1d(time_raw, gauge_raw, kind='linear', fill_value='extrapolate')
    interp_insp = interp1d(time_raw, insp_raw, kind='linear', fill_value='extrapolate')

    gauge_resampled = interp_gauge(t_new)
    insp_resampled = interp_insp(t_new)

    return t_new, gauge_resampled, insp_resampled

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Applies a zero-phase Butterworth low-pass filter to the signal.
    """
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def detect_single_apnea(t, signal, expected_duration, window_sec=1.0, target_fs=250.0):
    """
    Detecta a região de apneia de forma robusta utilizando limiar adaptativo
    e respeitando as bordas reais de transição do sinal.
    """
    # Guard against completely empty arrays sneaking through
    if len(t) == 0 or len(signal) == 0:
        return None

    window_samples = int(window_sec * target_fs)
    df = pd.DataFrame({'signal': signal})
    
    df['std'] = (df['signal'].rolling(window_samples, center=True, min_periods=1)
                 .std().bfill().ffill())
    
    normal_breathing_std = np.percentile(df['std'].dropna(), 75)
    adaptive_threshold = normal_breathing_std * 0.25 
    
    adaptive_threshold = max(min(adaptive_threshold, 0.06), 0.015)
    
    low_std_mask = (df['std'] < adaptive_threshold).to_numpy()

    regions = []
    in_apnea = False
    start_idx = 0
    for i, is_low in enumerate(low_std_mask):
        if is_low and not in_apnea:
            in_apnea = True
            start_idx = i
        elif not is_low and in_apnea:
            in_apnea = False
            end_idx = i - 1
            duration = t[end_idx] - t[start_idx]
            if duration > 3.0:   
                regions.append((start_idx, end_idx, duration))
    if in_apnea:
        duration = t[-1] - t[start_idx]
        if duration > 3.0:
            regions.append((start_idx, len(low_std_mask)-1, duration))

    if not regions:
        return None

    diffs = [abs(dur - expected_duration) for (_, _, dur) in regions]
    best_idx = np.argmin(diffs)
    s_idx, e_idx, real_dur = regions[best_idx]

    start_time = t[s_idx]
    end_time = t[e_idx]

    apnea_type = f"{expected_duration}s apnea"
    return (start_time, end_time, apnea_type)