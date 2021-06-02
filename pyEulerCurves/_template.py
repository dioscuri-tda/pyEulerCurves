# -*- coding: utf-8 -*-
"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt

from .ecc_VR import compute_local_contributions
from .ecc_cubical import compute_cubical_contributions
from .ecc_utils import euler_characteristic_list_from_all


class ECC_from_pointcloud(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, epsilon=0, workers=1):
        self.epsilon = epsilon
        self.workers = workers

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        # compute the list of local contributions to the ECC
        contributions_list, self.number_of_simplices = compute_local_contributions(
            X, self.epsilon, self.workers
        )

        # returns the ECC
        return euler_characteristic_list_from_all(contributions_list)


class ECC_from_bitmap(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, periodic_boundary=False, workers=1):
        self.periodic_boundary = periodic_boundary
        self.workers = workers

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        # compute the list of local contributions to the ECC
        contributions_list, self.number_of_simplices = compute_local_contributions(
            X, self.epsilon, self.workers
        )

        contributions_list = compute_cubical_contributions(top_dimensional_cells=X.flatten(order='F'),
                                                           dimensions=list(X.shape),
                                                           workers=2)

        self.number_of_simplices = sum([2*n+1 for n in X.shape])

        # returns the ECC
        return euler_characteristic_list_from_all(contributions_list)
















def plot_euler_curve(e_list, with_lines=False, title=None):
    plt.figure()
    plt.scatter([f[0] for f in e_list], [f[1] for f in e_list])

    # draw horizontal and vertical lines b/w points

    if with_lines:
        for i in range(1, len(e_list)):
            plt.vlines(
                x=e_list[i][0],
                ymin=min(e_list[i - 1][1], e_list[i][1]),
                ymax=max(e_list[i - 1][1], e_list[i][1]),
            )
            plt.hlines(y=e_list[i - 1][1], xmin=e_list[i - 1][0], xmax=e_list[i][0])

    plt.xlabel("filtration")
    plt.ylabel("euler characteristic")
    plt.title(title)
