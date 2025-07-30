# -*- coding: utf-8 -*-
"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

import matplotlib.pyplot as plt

from .ecc_pointcloud import (
    compute_local_contributions_VR,
    compute_local_contributions_alpha,
)
from .ecc_cubical import compute_cubical_contributions
from .ecc_utils import euler_characteristic_list_from_all


class ECC_from_pointcloud(TransformerMixin, BaseEstimator):
    """
    Transformer that computes Euler Characteristic Curves (ECCs) from a point cloud
    using Vietoris-Rips or Alpha filtrations.

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

    complex_type : {'VR', 'alpha'}, default='VR'
        Type of simplicial complex used for ECC computation. Use 'VR' for Vietoris-Rips
        and 'alpha' for Alpha complex.

    dbg : bool, default=False
        If True, enables debug output for internal steps.

    measure_times : bool, default=False
        If True, records timing information for different steps of the computation.

    Attributes
    ----------
    n_features_ : int
        Number of features seen during `fit`.

    contributions_list : list of tuples (filtration, contribution)
        The Euler characteristic contributions for each filtration value.

    num_simplices_list : list of int
        Number of simplices used in each ECC computation.

    largest_dimension_list : list of int
        Largest homology dimension computed for each point cloud.

    times : list of float
        If `measure_times` is True, contains the durations of computations.

    num_simplices : int
        Total number of simplices.
    """

    def __init__(
        self,
        epsilon=0,
        max_dimension=-1,
        workers=1,
        complex_type="VR",
        dbg=False,
        measure_times=False,
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

        complex_type : {'VR', 'alpha'}, default='VR'
            Type of simplicial complex used to compute ECCs. Choose 'VR' for Vietoris-Rips
            or 'alpha' for Alpha complex.

        dbg : bool, default=False
            Enable debug output.

        measure_times : bool, default=False
            Enable timing measurement.
        """
        self.epsilon = epsilon
        self.max_dimension = max_dimension
        if complex_type not in ("VR", "alpha"):
            raise ValueError(
                "Invalid complex_type: {}. Must be 'VR' or 'alpha'.".format(
                    complex_type
                )
            )

        self.complex_type = complex_type
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
        if self.complex_type == "VR":
            (
                self.contributions_list,
                self.num_simplices_list,
                self.largest_dimension_list,
                self.times,
            ) = compute_local_contributions_VR(
                X,
                self.epsilon,
                self.max_dimension,
                self.workers,
                self.dbg,
                self.measure_times,
            )
            self.num_simplices = sum(self.num_simplices_list)

        elif self.complex_type == "alpha":
            self.contributions_list, self.num_simplices = (
                compute_local_contributions_alpha(X, self.dbg)
            )

        else:
            raise ValueError(
                "Invalid complex_type: {}. Must be 'VR' or 'alpha'.".format(
                    self.complex_type
                )
            )

        # returns the ECC
        return euler_characteristic_list_from_all(self.contributions_list)


class ECC_from_bitmap(TransformerMixin, BaseEstimator):
    """
    Transformer that computes Euler Characteristic Curves (ECCs) from bitmap data
    using cubical complexes.

    This class supports both single-parameter and multi-parameter filtrations
    and is compatible with scikit-learn pipelines.

    Parameters
    ----------
    multifiltration : bool, default=False
        If True, enables computation using multi-parameter filtration, the last
        dimension in the bitmap is assumed to be the number of filtration parameters.
        Otherwise, a single-parameter filtration is used.

    periodic_boundary : bool or list of bool, default=False
        Specifies whether periodic boundary conditions should be applied.
        - If a single boolean, the same condition is applied to all dimensions.
        - If a list, it must match the number of spatial dimensions in the bitmap.

    workers : int, default=1
        Number of worker processes to use for parallel computation.

    chunksize : int, default=10
        Chunk size used in the multiprocessing pool for load balancing.

    Attributes
    ----------
    n_features_ : int
        Number of features seen during `fit`.

    contributions_list : list of tuples (filtration, contribution)
        The list of contributions to the Euler characteristic computed from the bitmap.

    number_of_simplices : int
        Estimated number of simplices (cubical cells) based on the input shape.
    """

    def __init__(
        self,
        multifiltration=False,
        periodic_boundary=False,
        workers=1,
        slicesize=2,
        chunksize=10,
        OLD=True,
    ):
        self.multifiltration = multifiltration
        self.periodic_boundary = periodic_boundary
        self.workers = workers
        self.slicesize = slicesize
        self.chunksize = chunksize
        self.OLD = OLD

    def fit(self, X, y=None):
        """
        Fit the transformer on input bitmap data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

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
        Compute the Euler Characteristic Curve (ECC) from bitmap input data.

        Parameters
        ----------
        X : ndarray
            An input array representing one or more bitmap volumes, where the last
            axis corresponds to filtration values. The spatial structure should
            follow NumPy's axis ordering ([z, y, x]).

        Returns
        -------
        ecc : list of [float, int] or None
            For single-parameter filtrations (`multifiltration=False`), returns
            a list of [filtration_value, Euler_characteristic] pairs representing
            the ECC.
            For multi-parameter filtrations, returns `None`. The list of contributions
            can be found in `self.contributions_list`.

        Raises
        ------
        ValueError
            If the shape of `X` does not match the shape seen in `fit`, or if
            the length of `periodic_boundary` does not match the bitmap dimensions.
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
        if not self.multifiltration:
            # reshape, adding an axis
            X = np.expand_dims(X, axis=-1)
        bitmap_dim = list(X.shape)[
            :-1
        ]  # the last dimension is the number of filtration parameters
        # number of filtration parameters
        num_f = X.shape[-1]
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
            top_dimensional_cells=X.reshape(
                -1, num_f, order="C"
            ),  # flattens the pixels
            dimensions=bitmap_dim,
            periodic_boundary=bitmap_boundary,
            workers=self.workers,
            slicesize=self.slicesize,
            chunksize=self.chunksize,
            OLD=self.OLD,
        )

        self.number_of_simplices = sum([2 * n + 1 for n in X.shape])

        # for the one parameter case, returns the ECC
        # returns None in the multifiltration case
        if self.multifiltration:
            self.contributions_list = sorted(
                self.contributions_list, key=lambda x: x[0]
            )
            # can't easily compute the Euler characteristic curve
            # just store the list of contributions in self.contributions_list
            return None
        else:
            # sort the contributions lists and return the ECC
            # convert the filtration values from tuples of lenght 1 to scalars
            self.contributions_list = sorted(
                [[k[0], i] for k, i in self.contributions_list], key=lambda x: x[0]
            )
            return euler_characteristic_list_from_all(self.contributions_list)
