import constants


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np

import json
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

embeddings = {}
text_labels = []

for file_path in constants.file_paths:
    with open(file_path + "_tokens_", "r") as f:
        embeddings_objects = json.load(f)
        for emb_obj in embeddings_objects:
            embeddings[emb_obj["hash"]] = np.array(emb_obj["embeddings"])

    with open(file_path + "_labeled_.json", "r") as f:
        text_labels_file = json.load(f)
        for text_hash, doc_info in text_labels_file.items():
            labels_list = [label["system"] for label in doc_info["label"]]
            text_labels.append((text_hash, labels_list))



filtered_embeddings = []
filtered_text_labels = []

for sample_id, classes in text_labels:
    if classes:
        for label in classes:
            filtered_embeddings.append(embeddings[sample_id])
            filtered_text_labels.append(label)

label_encoder = LabelEncoder()

X = np.array(filtered_embeddings)

y = label_encoder.fit_transform(filtered_text_labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trainer = Trainer()
trainer.train(X_train, y_train)

predictions = trainer.predict(X_test)

for name, preds in predictions.items():
    accuracy, f1, recall = trainer.evaluate(y_test, preds)
    print(f"{name} accuracy: {accuracy}, f1 score: {f1}, recall: {recall}")
