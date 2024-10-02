from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier




from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from typing import Tuple

class Trainer:
    def __init__(self) -> None:
        self.models = {
            'dummy_mf': DummyClassifier(strategy="most_frequent"),
            'dummy_un': DummyClassifier(strategy="uniform"),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'knn': KNeighborsClassifier(),
            'mlp': MLPClassifier(max_iter=1000),
        }

        self.param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'knn': {
                'n_neighbors': [3, 5, 7],
                'p': [1, 2]  
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd']
            },
            'dummy_mf': {},
            'dummy_pr': {},
            'dummy_st': {},
            'dummy_un': {},
        }
        self.trained_models = {}
        self.best_params = {}

    def train(self, X_train, y_train):
        for name, model in self.models.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=self.param_grids[name], cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.trained_models[name] = grid_search.best_estimator_
            self.best_params[name] = grid_search.best_params_

    def get_trained_models(self):
        return self.trained_models

    def predict(self, X_test):
        predictions = {}
        for name, model in self.trained_models.items():
            predictions[name] = model.predict(X_test)
        return predictions

    def evaluate(self, y_true, y_pred) -> Tuple[float, float, float]:
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        return accuracy, f1_weighted, recall_weighted, precision_weighted, f1_macro, recall_macro, precision_macro, f1_micro, recall_micro, precision_micro,

    def get_best_params(self):
        return self.best_params