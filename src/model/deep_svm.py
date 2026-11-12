"""
DeepSVM (Layered RBF-Kernel SVM Ensemble Classifier)

Based on Section 3.3 & Algorithm 2 of:
"Can You Understand Why I Am Crying? A Decision-making System for Classifying Infants' Cry Languages Based on DeepSVM Model"
(Rezaee et al., ACM 2024)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV


class DeepSVM:
    """
    Multi-layer RBF-kernel Support Vector Machine ensemble classifier.
    """

    def __init__(self, n_estimators=10, C=2.0, gamma="scale", random_state=42):
        self.n_estimators = n_estimators
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        base_svc = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
        calibrated_svc = CalibratedClassifierCV(estimator=base_svc, ensemble=False)

        self.layer1_ensemble = BaggingClassifier(
            estimator=calibrated_svc,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        top_svc = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
        self.layer2_classifier = CalibratedClassifierCV(estimator=top_svc, ensemble=False)

    def fit(self, X, y):
        self.layer1_ensemble.fit(X, y)
        l1_probs = self.layer1_ensemble.predict_proba(X)
        X_stacked = np.hstack([X, l1_probs])
        self.layer2_classifier.fit(X_stacked, y)
        return self

    def predict(self, X):
        l1_probs = self.layer1_ensemble.predict_proba(X)
        X_stacked = np.hstack([X, l1_probs])
        return self.layer2_classifier.predict(X_stacked)

    def predict_proba(self, X):
        l1_probs = self.layer1_ensemble.predict_proba(X)
        X_stacked = np.hstack([X, l1_probs])
        return self.layer2_classifier.predict_proba(X_stacked)
