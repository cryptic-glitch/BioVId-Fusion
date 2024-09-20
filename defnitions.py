from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
MODELS = {"svm": SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42, probability=True),
       "random_forest": RandomForestClassifier(max_depth=20)}
