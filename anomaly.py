import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationDetector:
    def __init__(self, contamination=0.05, random_state=0):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, features):
        self.model.fit(features)

    def is_ood(self, x):
        return self.model.predict(np.array(x).reshape(1, -1))[0] == -1
