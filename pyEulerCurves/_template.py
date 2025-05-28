# -*- coding: utf-8 -*-
"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

import matplotlib.pyplot as plt

from .ecc_VR import compute_local_contributions
from .ecc_cubical import compute_cubical_contributions
from .ecc_utils import euler_characteristic_list_from_all


class ECC_from_pointcloud(TransformerMixin, BaseEstimator):
    """
    Transformer that computes Euler Characteristic Curves (ECCs) from a point cloud
    using Vietoris-Rips filtrations.

    This transformer is compatible with scikit-learn pipelines and computes local
    contributions to the Euler characteristic, assembling them into a global ECC.

    Parameters
    ----------
    epsilon : float, default=0
        Threshold parameter for Vietoris-Rips filtration. Controls the scale at which
        simplices are created.

    max_dimension : int, default=-1
        Maximum homology dimension to consider. If set to -1, all dimensions are used.

    workers : int, default=1
        Number of worker processes to use in parallel computation.

    dbg : bool, default=False
        If True, enables debug output for internal steps.

    measure_times : bool, default=False
        If True, records timing information for different steps of the computation.

    Attributes
    ----------
    n_features_ : int
        Number of features seen during `fit`.

    contributions_list : list of np.ndarray
        Local Euler characteristic contributions for each sample.

    num_simplices_list : list of int
        Number of simplices used in each ECC computation.

    largest_dimension_list : list of int
        Largest homology dimension computed for each point cloud.

    times : list of float
        If `measure_times` is True, contains the durations of computations.

    num_simplices : int
        Total number of simplices over all samples.
    """

    def __init__(
        self, epsilon=0, max_dimension=-1, workers=1, dbg=False, measure_times=False
    ):
        """
        Initialize the ECC_from_pointcloud transformer.

        Parameters
        ----------
        epsilon : float, default=0
            Vietoris-Rips filtration scale parameter.

        max_dimension : int, default=-1
            Maximum homology dimension to consider.

        workers : int, default=1
            Number of parallel workers.

        dbg : bool, default=False
            Enable debug output.

        measure_times : bool, default=False
            Enable timing measurement.
        """
        self.epsilon = epsilon
        self.max_dimension = max_dimension
        self.workers = workers
        self.dbg = dbg
        self.measure_times = measure_times

    def fit(self, X, y=None):
        """
        Fit the transformer on input data `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            Ignored. This parameter exists for compatibility with
            scikit-learn pipelines.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = check_array(X, accept_sparse=True, allow_nd=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """
        Compute the Euler Characteristic Curve (ECC) for the given point cloud(s).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input point cloud data. Each row corresponds to a single point cloud.

        Returns
        -------
        ecc : list of [float, int]
            A list of `[filtration_value, Euler_characteristic]` pairs representing
            the Euler Characteristic Curve computed from the entire dataset.
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
        (
            self.contributions_list,
            self.num_simplices_list,
            self.largest_dimension_list,
            self.times,
        ) = compute_local_contributions(
            X,
            self.epsilon,
            self.max_dimension,
            self.workers,
            self.dbg,
            self.measure_times,
        )

        self.num_simplices = sum(self.num_simplices_list)

        # returns the ECC
        return euler_characteristic_list_from_all(self.contributions_list)


class ECC_from_bitmap(TransformerMixin, BaseEstimator):
    """
    Transformer that computes the Euler Characteristic Curve (ECC) from a bitmap image
    using cubical complexes.

    This transformer is compatible with scikit-learn pipelines and interprets the input
    as a binary or grayscale bitmap, computing the ECC based on connected components
    formed by cubical cells (voxels or pixels).

    Parameters
    ----------
    periodic_boundary : bool or list of bool, default=False
        Specifies whether to use periodic boundary conditions in each spatial dimension.
        If a list is provided, it must match the number of dimensions of the bitmap.

    workers : int, default=1
        Number of parallel workers to use for computing local contributions.

    Attributes
    ----------
    n_features_ : int
        Number of features (flattened bitmap size) seen during `fit`.

    contributions_list : list of np.ndarray
        List of local Euler characteristic contributions computed from the bitmap.

    number_of_simplices : int
        Estimated total number of cells (0D to top-dimensional) in the cubical complex.
    """

    def __init__(self, periodic_boundary=False, workers=1):
        """
        Initialize the ECC_from_bitmap transformer.

        Parameters
        ----------
        periodic_boundary : bool or list of bool, default=False
            Whether to apply periodic boundary conditions in each dimension.

        workers : int, default=1
            Number of parallel workers.
        """
        self.periodic_boundary = periodic_boundary
        self.workers = workers

    def fit(self, X, y=None):
        """
        Fit the transformer to the input bitmap.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples, typically flattened bitmap arrays.

        y : None
            Ignored. This parameter exists for compatibility with
            scikit-learn pipelines.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = check_array(X, accept_sparse=True, allow_nd=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """
        Compute the Euler Characteristic Curve (ECC) from bitmap images.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Flattened input bitmaps. Each row is treated as a bitmap image.

        Returns
        -------
        ecc : list of [float, int]
            A list of `[filtration_value, Euler_characteristic]` pairs representing
            the ECC computed from the bitmap.
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

        self.contributions_list = compute_cubical_contributions(
            top_dimensional_cells=X.flatten(order="C"),
            dimensions=bitmap_dim,
            periodic_boundary=bitmap_boundary,
            workers=2,
        )

        self.number_of_simplices = sum([2 * n + 1 for n in X.shape])

        # returns the ECC
        return euler_characteristic_list_from_all(self.contributions_list)
