from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
import numpy as np

class MLPWithCoef(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            model: MLPClassifier,
            **kwargs
    ):
        self.coef_ = None
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y.ravel())
        # Use the weights from the input layer as a proxy for feature importance
        first_layer_weights = self.model.coefs_[0]
        self.coef_ = np.mean(np.abs(first_layer_weights), axis=1).reshape(1, -1)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
