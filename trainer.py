from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score


from typing import Tuple
class Trainer:
    def __init__(self) -> None:
        self.models = {
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'knn': KNeighborsClassifier()
        }
        self.trained_models = {}

    def train(self, X_train, y_train):
        for _, model in self.models.items():
            model.fit(X_train, y_train)

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


embeddings = []
labels = []

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

trainer = Trainer()
trainer.train(X_train, y_train)

predictions = trainer.predict(X_test)

for name, preds in predictions.items():
    accuracy, f1, recall = trainer.evaluate(y_test, preds)
    print(f"{name} accuracy: {accuracy}, f1 score: {f1}, recall: {recall}")
