

# Person-Independent Pain Intensity Recognition

## Project Description
 
This project is focused on person-independent pain intensity recognition using machine learning techniques. The main objective of this project is to use Multimodal fusion, to classify pain intensity from two different types of data sources i.e Biosensors Data and Videos. Two types of fusion techniques were implemented, which were Early Fusion and Late Fusion.     
The data was grouped (from both the data sources) into binary categories: 'no/mild pain' (0-2) and 'high pain' (3-4). 

The following ML algorithms were used:
- SVM
- Random Forest
- XGBoost with Random Forest

## Pre-Processing

The following pre-processing techniques were implemented for Videos: 
- Head Pose Estimation using Dlib:  Features extracted included fold intensity, eyebrow distance, eye closure, mouth height, yaw, pitch, roll, and translation (x, y, z).
- Nasolabial Fold Intensity: Analyzing the deepening of the nasolabial fold as a pain indicator.
- Low-Pass Filtering
- Temporal Derivatives
- Statistical Parameters Extraction: Mean, Median , Standard Deviation, Range, Inter-quartile Range, Inter-decile Range and Median Absolute Deviation

The following pre-processing techniques were implemented for Biosignals:
- Filtering: A Butterworth bandpass filter was used to reduce noise and minimize the effects of trends in the signals.
- Noise Reduction for EMG: An additional noise reduction procedure based on Empirical Mode Decomposition (EMD) was applied to the EMG signals.
- Feature Extraction: Which include peak height, peak difference, mean absolute difference, Fourier coefficients, bandwidth. Additional features were
  derived based on entropy (approximate and sample entropy), stationarity, and statistical moments.
- Windowing: All features were computed on a window of 5.5 seconds, resulting in a total of 131 features extracted from the biosignals.

## Dataset
In these experiments the BioVid Heat Pain database [19] is analysed. It comprises 90 participants ((1) 18-35 years (n = 30 years; 15 men, 15 women), (2) 36-50
years (n = 30; 15 men, 15 women), and (3) 51-65 years (n = 30; 15 men, 15 women)) Each of the 4 diﬀerent stimulation strengths was applied 20 times to give rise to a total of 80 responses. During the experiments, high resolution video (from 3 diﬀerent cameras), sensor data of a Kinect, and a biophysiological ampliﬁer were recorded. The physiological channels included electromyography (EMG) (zygomaticus, corrugator and trapezius muscles), skin conductance level (SCL) and an electrocardiogram (ECG).

Results
## Results

| Method                        | My Results |
|-------------------------------|------------|
| Early Fusion SVM              | 0.66       |
| Early Fusion RF               | 0.67       |
| Early Fusion XGBoost          | 0.6975     |
| Video                         | 0.68       |
| Video (XGBoost)               | 0.7288     |
| Late Fusion (Weighted Average)| 0.66       |


## Project Structure

```
├── Pre-Processing/
│   ├── biosignal/
│   │   └── Biosignals_final.py  # Pre-processing script for biosignals data
│   ├── video/
│   │   ├── Complete Statistical Parameters.py  # Script for extracting statistical parameters from video data
│   │   ├── first second derivative.py  # Script for calculating derivatives
│   │   ├── low pass filter.py  # Low pass filter application
│   │   └── video_process.py  # Main video processing script
├── Data/
│   ├── Early Fusion/
│   │   └── merged.csv  # Merged dataset for early fusion
│   ├── Late Fusion/
│   │   ├── biosignals.csv  # Biosignals data for late fusion
│   │   └── video.csv  # Video data for late fusion
├── Utils/
│   └── utils.py  # Additional utility functions for the project
├── main.py  # Main script to run the project
├── definitions.py  # Definitions and configurations used across the project
├── __init__.py  # Initialization script for the package
├── README.md  # Project overview
└── Report.pdf  # Project Report

```
