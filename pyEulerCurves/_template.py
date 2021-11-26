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
    def __init__(self, epsilon=0, workers=1, dbg=False, measure_times=False):
        self.epsilon = epsilon
        self.workers = workers
        self.dbg = dbg
        self.measure_times = measure_times

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
        X = check_array(X, accept_sparse=True, allow_nd=True)

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
        X = check_array(X, accept_sparse=True, allow_nd=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        # compute the list of local contributions to the ECC
        (self.contributions_list, self.num_simplices_list,
        self.max_dimension_list,
        self.times) = compute_local_contributions(
            X, self.epsilon, self.workers,
            self.dbg, self.measure_times
        )

        self.num_simplices = sum(self.num_simplices_list)

        # returns the ECC
        return euler_characteristic_list_from_all(self.contributions_list)


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
        X = check_array(X, accept_sparse=True, allow_nd=True)

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
        X = check_array(X, accept_sparse=True, allow_nd=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        # compute the list of local contributions to the ECC
        # numpy array have the following dimension convention
        # [z,y,x] but we want it to be [x,y,z]
        bitmap_dim = list(X.shape)
        bitmap_dim.reverse()

        if type(self.periodic_boundary) is list:
            if len(self.periodic_boundary) != len(bitmap_dim):
                raise ValueError(
                    "Dimension of input is different from the number of boundary conditions"
                )
            bitmap_boundary = self.periodic_boundary.copy()
            bitmap_boundary.reverse()
        else:
            bitmap_boundary = False

        self.contributions_list = compute_cubical_contributions(top_dimensional_cells=X.flatten(order='C'),
                                                                dimensions=bitmap_dim,
                                                                periodic_boundary=bitmap_boundary,
                                                                workers=2)

        self.number_of_simplices = sum([2*n+1 for n in X.shape])

        # returns the ECC
        return euler_characteristic_list_from_all(self.contributions_list)
