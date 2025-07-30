import numpy as np


def difference_ECP(ecp_1, ecp_2, dims):
    """
    Compute the L1 difference between two 2D ECPs using an efficient prefix sum grid.

    Parameters
    ----------
    ecp_1 : list of ((x, y), value)
    ecp_2 : list of ((x, y), value)
    dims : tuple (x_min, x_max, y_min, y_max)
    return_contributions : bool
        Whether to return the combined and pruned contribution list

    Returns
    -------
    float or (float, contributions)
    """
    # Extract min and max values
    x_min, x_max, y_min, y_max = dims

    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Prune and finalize
    contributions = (
        [((x_min, y_min), 0)]
        + prune_contributions(contributions)
        + [((x_max, y_max), 0)]
    )

    # Extract sorted coordinate lists
    X_list = sorted(set([f[0] for f, c in contributions]))
    Y_list = sorted(set([f[1] for f, c in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}
    y_index = {y: i for i, y in enumerate(Y_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list), len(Y_list)), dtype=int)
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


from numba import njit


def difference_ECP_numba(ecp_1, ecp_2, dims):

    # Extract min and max values
    x_min, x_max, y_min, y_max = dims

    # Initialize contributions with corner points
    contributions = []
    contributions += ecp_1
    contributions += [(c[0], -c[1]) for c in ecp_2]

    # Prune and finalize
    contributions = (
        [((x_min, y_min), 0)]
        + prune_contributions(contributions)
        + [((x_max, y_max), 0)]
    )

    # Extract sorted coordinate lists
    X_list = sorted(set([f[0] for f, c in contributions]))
    Y_list = sorted(set([f[1] for f, c in contributions]))
    x_index = {x: i for i, x in enumerate(X_list)}
    y_index = {y: i for i, y in enumerate(Y_list)}

    # Fill sparse grid
    grid = np.zeros((len(X_list), len(Y_list)), dtype=int)
    print("creating ECP matrix of size {}".format(grid.shape))
    for (x, y), val in contributions:
        i = x_index[x]
        j = y_index[y]
        grid[i, j] += val

    ext = compute_prefix_sum(grid)
    difference = compute_difference(ext, X_list, Y_list)

    return difference


@njit
def compute_prefix_sum(grid):
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
def compute_difference(ext, X_list, Y_list):
    # Compute total difference using box volumes
    difference = 0
    for i in range(len(X_list) - 1):
        delta_x = X_list[i + 1] - X_list[i]
        for j in range(len(Y_list) - 1):
            delta_y = Y_list[j + 1] - Y_list[j]
            contribution = ext[i, j]
            difference += abs(contribution * delta_x * delta_y)
    return difference
