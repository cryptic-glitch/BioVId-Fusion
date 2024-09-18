import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy
from tqdm import tqdm  

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def simple_denoise(signal_data):
    window_size = 10
    return np.convolve(signal_data, np.ones(window_size) / window_size, mode='same')

def extract_features(signal_data, fs, signal_name):
    window_size = int(2.8 * fs) 
    features = []
    feature_names = []

    if len(signal_data) < window_size:
        print(f"Warning: Signal too short for full window extraction (Length: {len(signal_data)} samples)")
        return [np.nan] * 15, [f"{signal_name}_feature_{i}" for i in range(15)]

    for start in range(0, len(signal_data) - window_size + 1, window_size):
        window = signal_data[start:start + window_size]


        mean_val = np.mean(window)
        std_val = np.std(window)
        max_val = np.max(window)
        min_val = np.min(window)
        range_val = max_val - min_val
        mad_val = np.median(np.abs(window - np.median(window)))
        peak_height = np.max(window) - np.min(window)
        mean_abs_diff = np.mean(np.abs(np.diff(window)))
        fourier_coeffs = np.abs(np.fft.fft(window))[:5]
        approx_entropy = entropy(np.histogram(window, bins=10, density=True)[0])
        samp_entropy = entropy(np.histogram(np.diff(window), bins=10, density=True)[0])

        features.extend([
            mean_val, std_val, max_val, min_val, range_val, mad_val,
            peak_height, mean_abs_diff, *fourier_coeffs, approx_entropy, samp_entropy
        ])

        feature_names.extend([
            f"{signal_name}_mean", f"{signal_name}_std", f"{signal_name}_max", f"{signal_name}_min",
            f"{signal_name}_range", f"{signal_name}_mad", f"{signal_name}_peak_height",
            f"{signal_name}_mean_abs_diff",
            *[f"{signal_name}_fourier_{i+1}" for i in range(5)],
            f"{signal_name}_approx_entropy", f"{signal_name}_sample_entropy"
        ])

    return np.array(features), feature_names

def extract_ecg_features(ecg_signal, fs):
    if len(ecg_signal) < 2:
        print(f"Warning: ECG signal too short for QRS detection (Length: {len(ecg_signal)} samples)")
        return [np.nan, np.nan], ["ecg_mean_rr", "ecg_rmssd"]

    r_peaks, _ = find_peaks(ecg_signal, distance=fs / 2)
    rr_intervals = np.diff(r_peaks) / fs
    if len(rr_intervals) == 0:
        print("Warning: No RR intervals detected.")
        return [np.nan, np.nan], ["ecg_mean_rr", "ecg_rmssd"]

    mean_rr = np.mean(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return [mean_rr, rmssd], ["ecg_mean_rr", "ecg_rmssd"]

def preprocess_and_extract(emg_data, ecg_data, scl_data, fs):
    emg_filtered = butter_bandpass_filter(emg_data, 20, 250, fs)
    emg_cleaned = simple_denoise(emg_filtered)
    emg_features, emg_feature_names = extract_features(emg_cleaned, fs, "emg_trapezius")

    ecg_filtered = butter_bandpass_filter(ecg_data, 0.1, 250, fs)
    ecg_features, ecg_feature_names = extract_ecg_features(ecg_filtered, fs)

    scl_filtered = butter_bandpass_filter(scl_data, 0.05, 1, fs)
    scl_features, scl_feature_names = extract_features(scl_filtered, fs, "scl")

    combined_features = np.concatenate([emg_features, ecg_features, scl_features])
    combined_feature_names = emg_feature_names + ecg_feature_names + scl_feature_names

    return combined_features, combined_feature_names

def process_folder_structure(main_folder, output_folder, fs=1000):
    for subdir, dirs, files in os.walk(main_folder):
        csv_files = [file for file in files if file.endswith('.csv')]
        if csv_files:
            print(f"Processing folder: {subdir}")
            for file in tqdm(csv_files, desc=f"Processing {os.path.basename(subdir)}"):
                file_path = os.path.join(subdir, file)

                try:
                    data = pd.read_csv(file_path, sep='\t')
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                required_columns = {'gsr', 'ecg', 'emg_trapezius'}
                if not required_columns.issubset(data.columns):
                    print(f"Error: Missing required columns in {file_path}")
                    continue

                emg_data = data['emg_trapezius'].values
                ecg_data = data['ecg'].values
                scl_data = data['gsr'].values

                features, feature_names = preprocess_and_extract(emg_data, ecg_data, scl_data, fs)

                relative_path = os.path.relpath(subdir, main_folder)
                output_subdir = os.path.join(output_folder, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                                output_file_path = os.path.join(output_subdir, file)
                features_df = pd.DataFrame([features], columns=feature_names)
                features_df.to_csv(output_file_path, index=False)


main_folder = "/kaggle/input/biosignals-filtered/biosignals_filtered"
output_folder = "/kaggle/working/"
process_folder_structure(main_folder, output_folder)
