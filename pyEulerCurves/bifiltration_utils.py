import numpy as np
from numba import njit, jit

import matplotlib.pyplot as plt


def create_ecp_grid(contributions, dims, verbose=False):

    # Add dummy min and max corners
    min_corner = tuple(dim[0] for dim in dims)
    max_corner = tuple(dim[1] for dim in dims)
    if verbose:
        print("creating the ECP in the range {} - {}".format(min_corner, max_corner))
    contributions = [(min_corner, 0)] + contributions + [(max_corner, 0)]

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

    return ext, X_list, Y_list


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


def plot_ECP_NEW(
    contributions, dims, this_ax=None, colorbar=False, verbose=False, **kwargs
):

    if this_ax == None:
        this_ax = plt.gca()

    Z, f1_list, f2_list = create_ecp_grid(contributions, dims, verbose)
    Z = Z.T  # for compatibility with pcolormesh

    print(Z.shape, len(f1_list), len(f2_list))

    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z[:-1, :-1], **kwargs)

    this_ax.set_xlabel("Filtration 1")
    this_ax.set_ylabel("Filtration 2")

    if colorbar:
        plt.colorbar(im, ax=this_ax)

    return this_ax, Z, f1_list, f2_list


#################
# OLD FUNCTIONS #
#################


def EC_at_bifiltration(contributions, f1, f2):
    return sum([c[1] for c in contributions if (c[0][0] <= f1) and (c[0][1] <= f2)])


def plot_ECP_OLD(contributions, dims, this_ax=None, colorbar=False, **kwargs):

    f1min, f1max = dims[0]
    f2min, f2max = dims[1]

    if this_ax == None:
        this_ax = plt.gca()

    f1_list = [f1min] + sorted(set([c[0][0] for c in contributions])) + [f1max]
    f2_list = [f2min] + sorted(set([c[0][1] for c in contributions])) + [f2max]

    Z = np.zeros((len(f2_list), len(f1_list)))

    print(Z.shape, len(f1_list), len(f2_list))

    for i, f1 in enumerate(f1_list):
        for j, f2 in enumerate(f2_list):
            Z[j, i] = EC_at_bifiltration(contributions, f1, f2)

    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z[:-1, :-1], **kwargs)

    this_ax.set_xlabel("Filtration 1")
    this_ax.set_ylabel("Filtration 2")

    if colorbar:
        plt.colorbar(im, ax=this_ax)

    return this_ax, Z, f1_list, f2_list
