from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier




from sklearn.metrics import accuracy_score, f1_score, recall_score

from typing import Tuple

class Trainer:
    def __init__(self) -> None:
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5),
            'svm': SVC(),
            'knn': KNeighborsClassifier()
        }
        self.trained_models = {}

    def train(self, X_train, y_train):
        for name, model in self.models.items():
            self.trained_models[name] = model.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_test)
        return predictions

    def evaluate(self, y_true, y_pred) -> Tuple[float, float, float]:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        return accuracy, f1, recall