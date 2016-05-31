
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd


class CX(BaseEstimator, TransformerMixin):
    """
    CX Decomposition

    CX Decomposition is a low-rank approximation method that reduces the data
    to a small subset of the columns of the original matrix. Because the
    resulting low-rank approximation consists of actual columns of the original
    matrix interpretation is easier than PCA or SVD. The columns are selected
    according to their statistical leverage.

    TODO: Add citation.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X):
        n_rows, n_columns = X.shape
        k = self.columns

        # Find SVD
        U, S, V = randomized_svd(X, n_components=k)

        # Compute sampling probabilities
        probs = (1 / k) * np.power(np.linalg.norm(V, ord=2, axis=0), 2)

        # Create C matrix
        it = np.random.choice(n_columns, k, replace=True, p=probs)

        # Store sampled column indices and probabilities
        self.column_probs_ = probs
        self.sampled_columns_ = it

        return self

    def transform(self, X):
        it = self.sampled_columns_
        probs = self.column_probs_
        k = self.columns

        C = X[:, it] / np.sqrt(k**probs[it])
        return C
