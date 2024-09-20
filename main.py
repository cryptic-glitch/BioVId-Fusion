import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
from utils import Classifier

if __name__ == "__main__":
    # late fusion
    video_clf = Classifier(csv_path= "...", clf = "svm")
    y_pred_video, y_proba_video, y_test_video = video_clf.predict()
    print("video acc::", accuracy_score(y_pred_video, y_test_video))
    bio_clf = Classifier(csv_path= "...", clf='svm')
    y_pred_bio, y_proba_bio, y_test_bio = bio_clf.predict()
    print("bio acc::", accuracy_score(y_pred_bio, y_test_bio))
    assert all(y_test_bio == y_test_video), "Split is not uniform!"
    # --> weighted average
    print("late fusion acc::", accuracy_score(np.where(np.argmax((0.7 * y_proba_video + 0.3 * y_proba_bio) / 2, axis=-1) == 0, 0, 1), y_test_video))

    # early fusion
    combined_clf = Classifier(csv_path="...", clf = "svm")
    y_pred_combined, y_proba_combined, y_test_combined = combined_clf.predict()
    print("early fusion acc::", accuracy_score(y_pred_combined, y_test_combined))

    #hybrid
    early_fusion = (0.7 * y_proba_video + 0.3 * y_proba_bio) / 2
    late_fusion = y_proba_combined
    print("hybrid acc::", accuracy_score(np.where(np.argmax((early_fusion + late_fusion) / 2, axis=-1) == 0, 0, 1), y_test_video))