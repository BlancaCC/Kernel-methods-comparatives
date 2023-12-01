#####################################################################################
# RFF Estimator with less variance and error than sklearn implementation
# Example of use:
# 
# >>> X = np.random.rand(10,5)
# >>> n_components = 3
# >>> RFF = RFFEstimator(gamma=0.5, n_components=n_components)
# >>> RFF.fit(X)
# >>> RFF.transform(X)
#
#####################################################################################
import scipy.sparse as sp
import numpy as np

from sklearn.kernel_approximation import RBFSampler, check_random_state,  check_is_fitted, safe_sparse_dot

class RFFEstimator(RBFSampler):
     
    def fit(self, X, y=None):
       
        X = self._validate_data(X, accept_sparse="csr")
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        sparse = sp.issparse(X)
        if self.gamma == "scale":
            # var = E[X^2] - E[X]^2 if sparse
            X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
            self._gamma = 1.0 / (n_features * X_var) if X_var != 0 else 1.0
        else:
            self._gamma = self.gamma
        n_of_weights = -(-self.n_components//2)
        self.random_weights_ = (2.0 * self._gamma) ** 0.5 * random_state.normal(
            size=(n_features, n_of_weights)
        )

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)

        projections = np.concatenate((np.cos(projection), np.sin(projection)), axis=1)
        projections *= (2.0 / self.n_components) ** 0.5
        return projections
