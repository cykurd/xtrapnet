import torch
import numpy as np
from scipy.spatial import KDTree

class XtrapController:
    def __init__(self, trained_model, train_features, train_labels, mode='warn', backup_model=None):
        self.model = trained_model
        self.backup_model = backup_model
        self.mode = mode
        self.train_features = train_features
        self.train_labels = train_labels

        with torch.no_grad():
            preds = self.model.predict(self.train_features)
        self.min_pred = np.min(preds)
        self.max_pred = np.max(preds)

        self.kdtree = KDTree(self.train_features)

    def is_ood(self, x):
        return x[0] < 0 and x[1] > 0  # Example condition

    def predict(self, features):
        features = np.array(features)
        results = []
        for row in features:
            if self.is_ood(row):
                if self.mode == 'clip':
                    raw = self.model.predict(row.reshape(1, -1))[0][0]
                    res = np.clip(raw, self.min_pred, self.max_pred)
                elif self.mode == 'nearest_data':
                    _, idx = self.kdtree.query(row)
                    res = self.model.predict(self.train_features[idx].reshape(1, -1))[0][0]
                elif self.mode == 'symmetry':
                    sym_row = np.array([abs(row[0]), row[1]])
                    res = self.model.predict(sym_row.reshape(1, -1))[0][0]
                elif self.mode == 'warn':
                    print(f"Warning: OOD input {row}, returning model prediction anyway.")
                    res = self.model.predict(row.reshape(1, -1))[0][0]
                elif self.mode == 'zero':
                    res = 0.0
                elif self.mode == 'error':
                    raise ValueError(f"Error: OOD input {row}, cannot predict.")
                elif self.mode == 'highest_confidence':
                    pred, var = self.model.predict(row.reshape(1, -1), mc_dropout=True)
                    res = pred[np.argmin(var)]
                elif self.mode == 'backup':
                    if self.backup_model:
                        res = self.backup_model.predict(row.reshape(1, -1))[0][0]
                    else:
                        raise ValueError("Backup model not provided!")
                else:
                    res = self.model.predict(row.reshape(1, -1))[0][0]
            else:
                res = self.model.predict(row.reshape(1, -1))[0][0]
            results.append(res)
        return np.array(results)
