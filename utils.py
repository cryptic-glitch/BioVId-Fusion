from typing import Any

from sklearn.metrics import accuracy_score
import joblib
from defnitions import MODELS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Classifier():
    def __init__(self, clf: Any, csv_path: str = "", model_path: str = ""):
        if not model_path:
            # train the model from scratch
            assert csv_path.endswith(".csv"), "Please provide a valid csv file to start with!"
            self.data = pd.read_csv(csv_path)
            self.clf = MODELS.get(clf)
        else:
            # load the pre-trained
            self.clf = joblib.load(self.model_path)

    def predict(self):
        x_test, y_test = self._run(self.data)
        return self.clf.predict(x_test), self.clf.predict_proba(x_test), y_test

    def accuracy(self, y_test,y_pred):
        return accuracy_score(y_test, y_pred)

    def _preprocess(self, df):
        scaler = StandardScaler()
        # drop columns of other dataset
        return scaler.fit_transform(df.drop(columns=['filename', 'pain_intensity']))

    def _run(self, df):
        df['pain_intensity'] = df['pain_intensity'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
        y = df["pain_intensity"]
        x_preprocessed = self._preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=0.3, random_state=42)
        self.clf.fit(X_train, y_train)
        return X_test, y_test
