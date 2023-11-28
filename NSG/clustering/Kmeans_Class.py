from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal as mvn
import numpy as np


class Kmeans():

    """
    Standard k-means algorithm.

    Parameters
    ----------
        X : input data to be clustered
        C : cluster centers
        N : no. input data points
        K : no. clusters
        Dx : dimension of inputs

    Updates
    -------
        Dec 2020.

        Numerical errors can arise if no points are assigned to a
        cluster centre. Now, if it happens, that cluster centre
        is randomly reposition based on the statistics of X.
        Specifically, it is repositioned by sampling from a Gaussian
        with mean and cov equal to the mean and covariance matrix of X.

    Authors
    -------
        P.L.Green and Rimvydas Partauskas

    """

    def __init__(self, X, C, N, K, Dx):

        self.X = X
        self.C = C
        self.N = N
        self.K = K

        # Calculate mean and covariance matrix of the input
        # and check that they are the correct size.
        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X.T)
        assert len(X_mean) == Dx
        assert np.shape(X_cov)[0] == Dx
        assert np.shape(X_cov)[1] == Dx
        self.pX = mvn(X_mean, X_cov)

    def update_labels(self):
        """ We apply this after the clusters have moved / been initialised
        to update the labels (Z).

        The matrix M stores values of the squared distances between each data
        point and its assigned centroid, multiplied by its indicator.

        """

        # Set all labels to zero
        self.Z = np.zeros([self.N, self.K])

        # Matrix of distances between each point and each cluster
        self.D = distance_matrix(self.X, self.C)

        # Assign points to clusters
        for i in range(self.N):
            j_min = np.where(self.D[i, :] == np.min(self.D[i, :]))
            self.Z[i, j_min] = 1

    def train(self, N_itt):
        self.J_values = []
        """ Train kmeans algorithm over N_itt iterations.

        """
        # Loop over N_itt iterations
        for n_itt in range(N_itt):
            J = 0
            self.update_labels()
            for k in range(self.K):
                i = np.where(self.Z[:, k] == 1)
                if len(i[0]) == 0:
                    print('Iteration', n_itt,
                          'cluster', k,
                          'assigned to zero points. Repositioned at random.')
                    self.C[k] = self.pX.rvs()
                else:
                    self.C[k] = np.mean(self.X[i], 0)
                for j in range(self.N):
                    J = np.square(self.D[j, k] * self.Z[j, k])+J
            self.J_values.append(J)
        self.update_labels()
