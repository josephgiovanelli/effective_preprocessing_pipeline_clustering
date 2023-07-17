# Author: Lars Buitinck
# License: 3-clause BSD
from numbers import Real

import numpy as np
import pandas as pd


from fsfc.base import BaseFeatureSelector


class PearsonThreshold(BaseFeatureSelector):
    """Feature selector that removes all correlated features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, default=0
        Maximum correlation allowed, remove the features that have the same value or greater.

    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.features = None
        self.original_features = None

    def fit(self, X, y=None):
        """Compability for supervised techniques.
        """
        df = pd.DataFrame(X)
        corr = abs(df.corr())
        self.features = range(X.shape[-1])
        self.original_features = range(X.shape[-1])
        for i in range(X.shape[-1]):
            if i in self.features:
                to_drop = [idx for idx, elem in enumerate(corr.iloc[:, i] > self.threshold) if elem and idx != i]
                self.features = [elem for elem in self.features if elem not in to_drop]
        return self

    def _get_support_mask(self):
        return [elem in self.features for elem in self.original_features]

    def transform(self, X, y=None):
        return X[:, self.features]
