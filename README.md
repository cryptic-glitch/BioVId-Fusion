Person-Independent Pain Intensity Recognition
Project Description
This project is focused on person-independent pain intensity recognition using machine learning techniques. The objective is to classify pain intensity from video and biosensor data, particularly aiming for robust classification across various subjects. The data is grouped into binary categories: 'no/mild pain' (0-2) and 'high pain' (3-4), using models such as Random Forest, SVM, and Multilayer Perceptron.

The project employs both early and late fusion techniques to combine data from multiple sources, specifically biosensors and video, allowing for a more comprehensive analysis of pain intensity levels.

Features
Binary classification of pain intensity from multi-modal data sources (biosensors and video).
Implements various machine learning models including:
Random Forest
Support Vector Machine (SVM)
Multilayer Perceptron (MLP)
Uses early, late, and hybrid fusion methods to combine different data modalities.
Hyperparameter tuning for optimized classification accuracy.

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
└── requirements.txt  # Dependencies for the project
