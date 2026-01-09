import itertools
from collections import defaultdict

import numpy as np
from numba import njit


#######################################
############ NAIVE ####################
#######################################
def EC_at_value(contributions, *fs):
    """
    Calculate the sum of contributions for a given set of coordinates in n-dimensional space.

    Parameters
    ----------
    contributions : list of tuples
        A list where each element is a tuple containing a coordinate and a contribution value.

    *fs : float
        A variable number of threshold values representing the coordinates for each dimension.
        The number of values must match the number of dimensions of the coordinates in the contributions.

    Returns
    -------
    float
        The sum of contributions that are less than or equal to the given thresholds for each dimension.
    """
    return sum(
        [c[1] for c in contributions if all(c[0][i] <= fs[i] for i in range(len(fs)))]
    )


def difference_ECP_naive(ecp_1, ecp_2, dims, verbose=False):
    """
    Calculate the difference in contributions between two ECPs, given as list of contributions
    for an arbitrary number of dimensions.

    Parameters
    ----------
    ecp_1 : list of tuples
        A list of tuples, where each tuple contains a coordinate (in any dimension) and a contribution value.

    ecp_2 : list of tuples
        A list of tuples, where each tuple contains a coordinate (in any dimension) and a contribution value.

    dims : tuple of pairs
        dims : tuple ((x_i_min, x_i_max) for each dimension i)

    Returns
    -------
    float
        The difference in the energy contributions, calculated by subtracting contributions from `ecp_2`
        from `ecp_1`, integrating over the space defined by `dims`.

    Notes
    -----
    The function assumes that the dimensions in the `dims` parameter are ordered in the form
    `((f0min, f0max), (f1min, f1max), ..., (fnmin, fnmax))`, where each pair of consecutive values corresponds
    to the min and max of a specific dimension. The contributions should have coordinates matching the number of
    dimensions in `dims`.

    Examples
    --------
    ecp_1 = [((1, 2, 3), 10), ((4, 5, 6), 20)]
    ecp_2 = [((2, 3, 4), 5), ((3, 4, 5), 15)]
    dims = ((0, 5), (0, 5), (0, 5))  # 3D space, with x, y, z ranging from 0 to 5
    result = difference_nd_ECP(ecp_1, ecp_2, dims)
    print(result)
    """

    num_dims = len(dims)

    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Add dummy min and max corners
    min_corner = tuple(dim[0] for dim in dims)
    max_corner = tuple(dim[1] for dim in dims)
    if verbose:
        print(
            "computing L1 distance in the range {} - {}".format(min_corner, max_corner)
        )
    contributions = (
        [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
    )

    # Generate sorted lists of unique coordinates for each dimension
    coordinate_lists = [
        sorted(set([c[0][i] for c in contributions])) for i in range(num_dims)
    ]

    # Initialize the difference variable
    difference = 0.0

    # Initialize indices for each dimension (we start from the first coordinate)
    indices = [0] * num_dims

    # List to store deltas for each dimension (initialized to None, we'll calculate them later)
    deltas = [None] * num_dims

    while True:
        # Calculate the deltas for the current indices, i.e., the differences between adjacent coordinates
        for i in range(num_dims):
            deltas[i] = (
                coordinate_lists[i][indices[i] + 1] - coordinate_lists[i][indices[i]]
            )

        # Calculate the coordinates corresponding to the current indices
        coords = tuple(coordinate_lists[i][indices[i]] for i in range(num_dims))

        # Compute the contribution at the current coordinates
        contribution_value = EC_at_value(contributions, *coords)

        # Multiply the contribution by the volume of the current "box" (product of deltas)
        difference += np.abs(contribution_value * np.prod(deltas))

        # Move to the next set of indices (advance the indices like counting in a multi-dimensional array)
        for i in range(
            num_dims - 1, -1, -1
        ):  # Start from the last dimension and move backwards
            indices[i] += 1
            if indices[i] < len(coordinate_lists[i]) - 1:
                break  # If we've not overflowed, stop incrementing
            else:
                indices[i] = 0  # Reset this dimension and move to the next dimension

        # If we've exhausted all indices, break out of the loop
        if all(idx == 0 for idx in indices):
            break

    return difference


#######################################
######### INCLUSION / EXCLUSION #######
#######################################


def difference_ECP_2d_IE(ecp_1, ecp_2, dims, verbose=False):
    """
    Compute the L1 difference between two 2D ECPs using an efficient prefix sum grid.

    Parameters
    ----------
    ecp_1 : list of ((x, y), value)
    ecp_2 : list of ((x, y), value)
    dims : tuple ((x_min, x_max), (y_min, y_max))
    return_contributions : bool
        Whether to return the combined and pruned contribution list

    Returns
    -------
    float or (float, contributions)
    """
    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Add dummy min and max corners
    min_corner = tuple(dim[0] for dim in dims)
    max_corner = tuple(dim[1] for dim in dims)
    if verbose:
        print(
            "computing L1 distance in the range {} - {}".format(min_corner, max_corner)
        )
    contributions = (
        [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
    )

    # Extract sorted coordinate lists
    X_list = sorted(set([f[0] for f, c in contributions]))
    Y_list = sorted(set([f[1] for f, c in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}
    y_index = {y: i for i, y in enumerate(Y_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list), len(Y_list)), dtype=int)
    if verbose:
        print("creating ECP matrix of size {}".format(grid.shape))
    for (x, y), val in contributions:
        i = x_index[x]
        j = y_index[y]
        grid[i, j] += val

    # Build prefix-sum extended grid
    ext = np.copy(grid)
    for i in range(ext.shape[0]):
        for j in range(ext.shape[1]):
            if i > 0:
                ext[i, j] += ext[i - 1, j]
            if j > 0:
                ext[i, j] += ext[i, j - 1]
            if i > 0 and j > 0:
                ext[i, j] -= ext[i - 1, j - 1]

    # Compute total difference using box volumes
    difference = 0
    for i in range(len(X_list) - 1):
        delta_x = X_list[i + 1] - X_list[i]
        for j in range(len(Y_list) - 1):
            delta_y = Y_list[j + 1] - Y_list[j]
            contribution = ext[i, j]
            difference += abs(contribution * delta_x * delta_y)

    return difference


def prune_contributions(contributions):
    """
    Prune contributions by summing values for each unique coordinate and removing
    any contributions that result in a total value of zero.

    Parameters
    ----------
    contributions : list of tuples
        A list where each element is a tuple containing a coordinate (tuple/list) and a contribution value.
        Coordinates may repeat, and the function will sum the contributions with the same coordinates.

    Returns
    -------
    list of tuples
        A sorted list of tuples, where each tuple contains a unique coordinate and its total contribution,
        excluding coordinates with a zero contribution.
    """
    total_ECP = dict()

    # Sum contributions for each unique coordinate
    for f, c in contributions:
        total_ECP[f] = total_ECP.get(f, 0) + c

    # Remove the contributions that are zero
    to_del = [key for key, value in total_ECP.items() if value == 0]
    for key in to_del:
        del total_ECP[key]

    # Return sorted list of tuples (coordinate, total contribution)
    return sorted(total_ECP.items(), key=lambda x: x[0])


#######################################
############ NUMBA ####################
#######################################


def difference_ECP_1d_numba(ecp_1, ecp_2, dims, verbose=False):
    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Add dummy min and max corners
    min_corner = (dims[0][0],)
    max_corner = (dims[0][1],)
    if verbose:
        print(
            "computing L1 distance in the range {} - {}".format(min_corner, max_corner)
        )
    contributions = (
        [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
    )

    # Extract sorted coordinate list
    X_list = sorted(set([f[0] for f, _ in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list),), dtype=int)
    if verbose:
        print("creating ECP matrix of size {}".format(grid.shape))
    for (x,), val in contributions:
        i = x_index[x]
        grid[i] += val

    difference = compute_difference_1d(grid, X_list)

    return difference


@njit
def compute_difference_1d(ext, X_list):

    for i in range(1, len(ext)):
        ext[i] += ext[i - 1]

    difference = 0
    for i in range(len(X_list) - 1):
        delta_x = X_list[i + 1] - X_list[i]
        difference += abs(ext[i] * delta_x)
    return difference


#######################################################


def difference_ECP_2d_numba(ecp_1, ecp_2, dims, verbose=False):

    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Add dummy min and max corners
    min_corner = tuple(dim[0] for dim in dims)
    max_corner = tuple(dim[1] for dim in dims)
    if verbose:
        print(
            "computing L1 distance in the range {} - {}".format(min_corner, max_corner)
        )
    contributions = (
        [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
    )

    # Extract sorted coordinate lists
    X_list = sorted(set([f[0] for f, c in contributions]))
    Y_list = sorted(set([f[1] for f, c in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}
    y_index = {y: i for i, y in enumerate(Y_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list), len(Y_list)), dtype=int)
    if verbose:
        print("creating ECP matrix of size {}".format(grid.shape))
    for (x, y), val in contributions:
        i = x_index[x]
        j = y_index[y]
        grid[i, j] += val

    ext = compute_prefix_sum_2d(grid)
    difference = compute_difference_2d(ext, X_list, Y_list)

    return difference


@njit
def compute_prefix_sum_2d(grid):
    # Build prefix-sum extended grid
    ext = np.copy(grid)
    for i in range(ext.shape[0]):
        for j in range(ext.shape[1]):
            if i > 0:
                ext[i, j] += ext[i - 1, j]
            if j > 0:
                ext[i, j] += ext[i, j - 1]
            if i > 0 and j > 0:
                ext[i, j] -= ext[i - 1, j - 1]
    return ext


@njit
def compute_difference_2d(ext, X_list, Y_list):
    # Compute total difference using box volumes
    difference = 0
    for i in range(len(X_list) - 1):
        delta_x = X_list[i + 1] - X_list[i]
        for j in range(len(Y_list) - 1):
            delta_y = Y_list[j + 1] - Y_list[j]
            contribution = ext[i, j]
            difference += abs(contribution * delta_x * delta_y)
    return difference


def difference_ECP_3d_numba(ecp_1, ecp_2, dims, verbose=False):
    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Add dummy min and max corners
    min_corner = tuple(dim[0] for dim in dims)
    max_corner = tuple(dim[1] for dim in dims)
    if verbose:
        print(
            "computing L1 distance in the range {} - {}".format(min_corner, max_corner)
        )
    contributions = (
        [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
    )

    # Extract sorted coordinate lists
    X_list = sorted(set([f[0] for f, c in contributions]))
    Y_list = sorted(set([f[1] for f, c in contributions]))
    Z_list = sorted(set([f[2] for f, c in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}
    y_index = {y: i for i, y in enumerate(Y_list)}
    z_index = {z: i for i, z in enumerate(Z_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list), len(Y_list), len(Z_list)), dtype=int)
    if verbose:
        print("creating ECP matrix of size {}".format(grid.shape))
    for (x, y, z), val in contributions:
        i = x_index[x]
        j = y_index[y]
        k = z_index[z]
        grid[i, j, k] += val

    ext = compute_prefix_sum_3d(grid)
    difference = compute_difference_3d(ext, X_list, Y_list, Z_list)

    return difference


@njit
def compute_prefix_sum_3d(grid):
    ext = np.copy(grid)
    nx, ny, nz = ext.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                total = ext[i, j, k]
                if i > 0:
                    total += ext[i - 1, j, k]
                if j > 0:
                    total += ext[i, j - 1, k]
                if k > 0:
                    total += ext[i, j, k - 1]
                if i > 0 and j > 0:
                    total -= ext[i - 1, j - 1, k]
                if i > 0 and k > 0:
                    total -= ext[i - 1, j, k - 1]
                if j > 0 and k > 0:
                    total -= ext[i, j - 1, k - 1]
                if i > 0 and j > 0 and k > 0:
                    total += ext[i - 1, j - 1, k - 1]
                ext[i, j, k] = total
    return ext


@njit
def compute_difference_3d(ext, X_list, Y_list, Z_list):
    difference = 0.0
    nx, ny, nz = ext.shape
    for i in range(nx - 1):
        delta_x = X_list[i + 1] - X_list[i]
        for j in range(ny - 1):
            delta_y = Y_list[j + 1] - Y_list[j]
            for k in range(nz - 1):
                delta_z = Z_list[k + 1] - Z_list[k]
                val = ext[i, j, k]
                difference += abs(val * delta_x * delta_y * delta_z)
    return difference


#######################################
############ ANY DIM ##################
#######################################


def difference_ECP(ecp_1, ecp_2, dims, verbose=False):
    """
    Compute the L1 difference between two Euler Characteristic Profiles (ECPs)
    over a given multi-dimensional space.

    This function automatically selects an optimized method based on the number of
    dimensions (1D, 2D, 3D), or falls back to a generic N-dimensional method
    using prefix sums and inclusion-exclusion for higher dimensions.

    Parameters
    ----------
    ecp_1 : list of tuples
        First ECP, as a list of (coordinate, value) tuples.
        Each coordinate is a tuple of floats with length equal to the number of dimensions.

    ecp_2 : list of tuples
        Second ECP, in the same format as `ecp_1`.

    dims : tuple of (min, max) pairs
        A tuple defining the lower and upper bounds for each dimension.
        Example for 3D: ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    verbose : bool, optional
        If True, prints debug output including the shape of the generated grid.
        Default is False.

    Returns
    -------
    float
        The L1 difference (i.e., total absolute difference in contribution values)
        between the two ECPs, integrated over the specified multi-dimensional space.

    Notes
    -----
    - For 1D, 2D, and 3D ECPs, optimized versions using Numba and prefix sums are used.
    - For N > 3 dimensions, a generic algorithm based on inclusion-exclusion is applied.
    - The input ECPs must be defined over the same dimensionality as `dims`.


    Example
    -------
    A 2D example comparing two ECPs over a square region:

    >>> ecp_1 = [((1, 1), 10), ((2, 2), 5)]
    >>> ecp_2 = [((1, 1), 3), ((3, 3), 7)]
    >>> dims = ((0, 4), (0, 4))  # x and y range from 0 to 4
    >>> difference_ECP(ecp_1, ecp_2, dims)
    120

    """

    ndim = len(dims)

    if ndim == 1:
        if verbose:
            print("using optimized numba 1d")
        return difference_ECP_1d_numba(ecp_1, ecp_2, dims, verbose)
    elif ndim == 2:
        if verbose:
            print("using optimized numba 2d")
        return difference_ECP_2d_numba(ecp_1, ecp_2, dims, verbose)
    elif ndim == 3:
        if verbose:
            print("using optimized numba 3d")
        return difference_ECP_3d_numba(ecp_1, ecp_2, dims, verbose)
    else:
        # Generic N-dimensional fallback
        contributions = []
        contributions += ecp_1
        contributions += [(f, -c) for f, c in ecp_2]

        min_corner = tuple(dim[0] for dim in dims)
        max_corner = tuple(dim[1] for dim in dims)
        contributions = (
            [(min_corner, 0)] + prune_contributions(contributions) + [(max_corner, 0)]
        )

        coord_lists = [
            sorted(set(pt[i] for pt, _ in contributions)) for i in range(ndim)
        ]
        index_maps = [
            {val: idx for idx, val in enumerate(axis)} for axis in coord_lists
        ]
        shape = tuple(len(axis) for axis in coord_lists)

        if verbose:
            print(f"Creating ECP matrix of size {shape}")
        grid = np.zeros(shape, dtype=int)

        for pt, val in contributions:
            idx = tuple(index_maps[i][pt[i]] for i in range(ndim))
            grid[idx] += val

        ext = compute_prefix_sum_nd(grid)
        difference = compute_difference_nd(ext, coord_lists)
        return difference


def compute_prefix_sum_nd(grid):
    """
    N-dimensional prefix sum using inclusion-exclusion.
    """
    ext = np.copy(grid)
    shape = ext.shape
    ndim = ext.ndim

    it = np.ndindex(shape)
    for idx in it:
        total = ext[idx]
        for axis in range(ndim):
            if idx[axis] == 0:
                continue
            prev_idx = list(idx)
            prev_idx[axis] -= 1
            total += ext[tuple(prev_idx)]
        # Subtract over-counted intersections
        for r in range(2, ndim + 1):
            for axes in itertools.combinations(range(ndim), r):
                skip = False
                prev_idx = list(idx)
                for ax in axes:
                    if prev_idx[ax] == 0:
                        skip = True
                        break
                    prev_idx[ax] -= 1
                if not skip:
                    sign = (-1) ** (r + 1)
                    total += sign * ext[tuple(prev_idx)]
        ext[idx] = total
    return ext


def compute_difference_nd(ext, coord_lists):
    """
    Computes N-dimensional difference using box volumes and prefix sum grid.

    Args:
        ext: N-dimensional prefix sum array
        coord_lists: List of sorted coordinate arrays for each dimension

    Returns:
        Scalar total weighted sum
    """
    shape = ext.shape
    total = 0

    it = np.ndindex(tuple(s - 1 for s in shape))
    for idx in it:
        # Compute the size of the hyperrectangle (cell volume)
        volume = 1.0
        for axis in range(ext.ndim):
            delta = coord_lists[axis][idx[axis] + 1] - coord_lists[axis][idx[axis]]
            volume *= delta

        val = ext[idx]
        total += abs(val * volume)

    return total


def discretize_contributions(contributions, dims, resolution, verbose=False):
    """
    Discretize a list of contributions onto a regular grid and return as a list of index-value pairs.

    Parameters
    ----------
    contributions : list of tuples
        Each element is a tuple ((x1, x2, ..., xn), value), where the coordinates
        represent a point in n-dimensional space and value is a scalar.

    dims : tuple of tuples
        Bounds for each dimension in the form ((x1_min, x1_max), ..., (xn_min, xn_max)).

    resolution : tuple of ints
        Number of grid cells in each dimension, must match the number of dimensions.

    verbose : bool
        Print where each value gets mapped to the grid. Default False.

    Returns
    -------
    list of tuples
        Each element is a tuple (index_tuple, value), where index_tuple is the
        grid index (e.g., (i, j) in 2D) and value is the sum of contributions to that cell.

    Example (2D)
    ------------
    >>> contributions = [((1.1, 2.2), 5), ((3.9, 1.5), 3)]
    >>> dims = ((0, 4), (0, 4))
    >>> resolution = (4, 4)
    >>> discretize_contributions(contributions, dims, resolution)
    [((1, 2), 5), ((3, 1), 3)]
    """
    ndim = len(dims)
    assert len(resolution) == ndim, "Resolution must match number of dimensions."

    cell_values = defaultdict(int)

    grid = [
        np.linspace(start=d[0], stop=d[1], num=resolution[i], endpoint=False)
        for i, d in enumerate(dims)
    ]

    for coords, value in contributions:
        new_coord = []
        for d in range(ndim):
            # right a[i-1] <= v < a[i]
            id = np.searchsorted(grid[d], coords[d], side="right") - 1

            if verbose:
                print("value {} goes to id {} - {}".format(coords[d], id, grid[d][id]))

            new_coord.append(grid[d][id])
        cell_values[tuple(new_coord)] += value

    return list(cell_values.items())
