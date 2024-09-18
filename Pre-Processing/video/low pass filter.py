from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd

def low_pass_filter(data, cutoff=0.1, order=3):
    # cutoff = frequency at which the filter begins to reduce the higher frequency components of the signal
    # range is around 0-0.5, A lower cutoff only allows lower frequency to pass through
    # low cut-off = more aggressive filter
    # 0.5 = Nyquist Frequency - Converts continuous function/ signal to discrete sequence

    # order = determines the sharpness of the cutoff. 3 is common
    b, a = butter(order, cutoff, btype='low', analog=False)  # Order and cutoff corrected
    filtered_data = filtfilt(b, a, data)
    # Clip any negative values to zero
    filtered_data = np.clip(filtered_data, 0, None)
    return filtered_data

df = pd.read_csv('/home/prashant/Desktop/all/Datasets/BioVid_pain/PartA/fianl dataset/Video_Signals/Video_Pain_Intensity_RAW.csv')
numeric_data = df.select_dtypes(include=['float64', 'int64']).columns.drop(['Frame', 'Pain Intensity'])

filtered_data = df.copy()
for col in numeric_data:
    filtered_data[col] = low_pass_filter(df[col])

filtered_data.to_csv('/home/prashant/Desktop/all/Datasets/BioVid_pain/PartA/fianl dataset/Video_Signals/fianl final/1.csv')
print("done")
