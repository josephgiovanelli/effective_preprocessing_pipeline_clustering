import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.multiclass import check_classification_targets
from imblearn.base import BaseSampler
from imblearn.utils import check_sampling_strategy


class MyOutlierDetector(BaseSampler):    

    _sampling_type = "bypass"

    def __init__(self, *, n_neighbors = 2, accept_sparse=True, kw_args=None, validate=True):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.estimator = LocalOutlierFactor(self.n_neighbors)
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.validate = validate

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.
        You should use ``fit_resample`` in all cases.
        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.
        y : array-like of shape (n_samples,)
            Target array.
        Returns
        -------
        self : object
            Return the instance itself.
        """
        # we need to overwrite SamplerMixin.fit to bypass the validation
        if self.validate:
            check_classification_targets(y)
            X, y, _ = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        return self

    def _fit_resample(self, X, y):
        filter = self.estimator.fit_predict(X)
        new_X = np.array([X[i, :] for i in range(len(filter)) if filter[i] != -1])
        new_y = np.array([y[i] for i in range(len(filter)) if filter[i] != -1])
        return new_X, new_y