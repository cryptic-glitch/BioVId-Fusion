import cv2
import dlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import gc
from tqdm import tqdm


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/kaggle/input/dlib-model-sid/shape_predictor_68_face_landmarks.dat")  # Ensure you have this model file

# Video input path and output CSV path
dataset_path = "/kaggle/input/biovid/video"  # Replace with your video path
output_csv = "/kaggle/working/combined_features"

size = (1000,1000)
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1))

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Function to calculate gradient magnitude in the nasolabial fold region
def calculate_nasolabial_fold_intensity(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.mean(gradient_magnitude)

# Function to extract facial expression features
def extract_features(landmarks):
    brow_left = (landmarks.part(19).x, landmarks.part(19).y)  # Left brow
    eye_left = (landmarks.part(37).x, landmarks.part(37).y)  # Left eye (top point)
    brow_right = (landmarks.part(24).x, landmarks.part(24).y)  # Right brow
    eye_right = (landmarks.part(44).x, landmarks.part(44).y)  # Right eye (top point)
    eye_brow_distance = (np.linalg.norm(np.array(brow_left) - np.array(eye_left)) +
                         np.linalg.norm(np.array(brow_right) - np.array(eye_right))) / 2
    eye_closure = np.linalg.norm(np.array((landmarks.part(38).x, landmarks.part(38).y)) -
                                 np.array((landmarks.part(40).x, landmarks.part(40).y)))
    mouth_top = (landmarks.part(51).x, landmarks.part(51).y)  # Top lip
    mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)  # Bottom lip
    mouth_height = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
    return eye_brow_distance, eye_closure, mouth_height


# Function to process each video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Nasolabial fold intensity
            top_left = (landmarks.part(31).x, landmarks.part(31).y)
            bottom_right = (landmarks.part(35).x, landmarks.part(48).y)
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            fold_intensity = calculate_nasolabial_fold_intensity(roi)

            # Facial expression features
            eye_brow_distance, eye_closure, mouth_height = extract_features(landmarks)

            # Head pose estimation
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
            yaw, pitch, roll = euler_angles.flatten()

            # Append all extracted data to results
            results.append([
                frame_count, fold_intensity, eye_brow_distance, eye_closure, mouth_height,
                yaw, pitch, roll, translation_vector[0][0], translation_vector[1][0], translation_vector[2][0]
            ])

            frame_count += 1

    cap.release()
    

    # cv2.destroyAllWindows()

    return results


video_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".mp4"):
            video_files.append(os.path.join(root, file))
batches_of_video_files = [video_files[x:x + 100] for x in range(0, len(video_files), 100)]


# results = []

def multi_process_video(video_files):
    results = []
    for video_path in tqdm(video_files, desc="Processing Video"):
        results.extend(process_video(video_path))
    return results


results = []
filenames_all = []
for each_file in tqdm(video_files):
    filename_each = each_file.split("/")[-1]
    filenames_all.append(filename_each)
    results.extend(process_video(each_file))

for ind, val in enumerate(results):
    val.insert(0, filenames_all[ind])

df = pd.DataFrame(results, columns=["FileNames",
    "Frame", "Nasolabial Fold Intensity", "Eye-Brow Distance", "Eye Closure", "Mouth Height",
    "Yaw", "Pitch", "Roll", "Translation_X", "Translation_Y", "Translation_Z"
])
df.to_csv(os.path.join(output_csv, "features.csv"), index=False)