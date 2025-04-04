import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA class.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the dataset.

        Parameters:
        X (numpy.ndarray): The dataset with shape (n_samples, n_features).
        """
        # Compute the mean of each feature
        self.mean = np.mean(X, axis=0)
        # Center the data
        X_centered = X - self.mean
        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        # Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Apply the PCA transformation to the dataset.

        Parameters:
        X (numpy.ndarray): The dataset with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed dataset with shape (n_samples, n_components).
        """
        if self.components is None:
            raise ValueError("The PCA model has not been fitted yet.")
        # Center the data
        X_centered = X - self.mean
        # Project the data onto the principal components
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the PCA model and apply the transformation.

        Parameters:
        X (numpy.ndarray): The dataset with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: The transformed dataset with shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)