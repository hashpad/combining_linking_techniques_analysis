from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier

import numpy as np



from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from typing import Tuple

class Trainer:
    def __init__(self) -> None:
        self.models = {
            'dummy_mf': DummyClassifier(strategy="most_frequent"),
            'dummy_un': DummyClassifier(strategy="uniform"),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(probability=True),
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

    
    # def precision_at_k(self, y_true, y_pred, k=2):
    #     if len(y_pred)>k:
    #         y_pred = y_pred[:k]

    #     score = 0.0
    #     num_hits = 0.0

    #     for i,p in enumerate(y_pred):
    #         if p in y_true and p not in y_pred[:i]:
    #             num_hits += 1.0
    #             score += num_hits / (i+1.0)

    #     if not y_true:
    #         return 0.0

    #     return score / min(len(y_true), k)

    # def precision_at_k(self, y_true, y_pred, k=2):
    #     total_precision = 0
    #     num_samples = len(y_true)
        
    #     for true_labels, pred_labels in zip(y_true, y_pred):
    #         top_k_pred = pred_labels[:k]
            
    #         relevant_in_top_k = len(set(top_k_pred) & set(true_labels))
            
    #         tp = relevant_in_top_k
    #         fp = len([x for x in top_k_pred if x not in set(true_labels)])
    #         if tp + fp == 0:
    #             precision = 0
    #         else:
    #             precision = tp / (tp + fp)
    #         total_precision += precision
    
    #     # avg
    #     precision_at_k = total_precision / num_samples
    #     return precision_at_k
    def precision_at_k(self, y_true, y_pred, k=2):
        total_precision = 0
        num_samples = len(y_true)
        
        for true_labels, pred_labels in zip(y_true, y_pred):
            top_k_pred = pred_labels[:k]
            
            relevant_in_top_k = len(set(top_k_pred) & set(true_labels))
            
            precision = relevant_in_top_k / k
            total_precision += precision
    
        # avg
        precision_at_k = total_precision / num_samples
        return precision_at_k

    def recall_at_k(self, y_true, y_pred, k):
        total_recall = 0
        num_samples = len(y_true)
        
        for true_labels, pred_labels in zip(y_true, y_pred):
            top_k_pred = pred_labels[:k]
            
            relevant_in_top_k = len(set(top_k_pred) & set(true_labels))
            
            recall = relevant_in_top_k / len(true_labels) if len(true_labels) > 0 else 0
            total_recall += recall
        
        # avg
        recall_at_k = total_recall / num_samples
        return recall_at_k


    def f1_at_k(self, y_true, y_pred, k):
        precision = self.precision_at_k(y_true, y_pred, k)
        recall = self.recall_at_k(y_true, y_pred, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1_at_k = 2 * (precision * recall) / (precision + recall)
        return f1_at_k


    def get_best_params(self):
        return self.best_params
    
    