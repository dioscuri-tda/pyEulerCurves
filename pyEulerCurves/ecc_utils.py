## various functions
from matplotlib import pyplot as plt


# given the ordered list of local contributions
# returns a list of tuples (filtration, euler characteristic)
def euler_characteristic_list_from_all(local_contributions):
    """
    Construct the Euler Characteristic Curve (ECC) from local contributions.

    Parameters
    ----------
    local_contributions : list of [float, int]
        A sorted list of pairs where each element is a [filtration_value, contribution].

    Returns
    -------
    ecc : list of [float, int]
        List of [filtration, Euler characteristic] values forming the ECC.
    """

    euler_characteristic = []
    old_f, current_characteristic = local_contributions[0]

    for filtration, contribution in local_contributions[1:]:
        if filtration > old_f:
            euler_characteristic.append([old_f, current_characteristic])
            old_f = filtration

        current_characteristic += contribution

    # add last contribution
    if len(local_contributions) > 1:
        euler_characteristic.append([filtration, current_characteristic])

    if len(local_contributions) == 1:
        euler_characteristic.append(local_contributions[0])

    return euler_characteristic


# In[6]:


# WARNING
# when plotting a lot of points, drawing the lines can take some time
def plot_euler_curve(e_list, this_ax=None, with_lines=False, **kwargs):
    """
    Plot an Euler Characteristic Curve (ECC).

    Parameters
    ----------
    e_list : list of [float, int]
        The ECC to plot, as a list of [filtration, Euler characteristic] pairs.

    this_ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, uses current Axes.

    with_lines : bool, default=False
        If True, draws step-style horizontal and vertical lines between points.

    **kwargs : dict
        Additional keyword arguments passed to matplotlib scatter.

    Returns
    -------
    this_ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    if this_ax is None:
        this_ax = plt.gca()

    # Plotting
    this_ax.scatter([f[0] for f in e_list], [f[1] for f in e_list])
    # draw horizontal and vertical lines b/w points
    if with_lines:
        for i in range(1, len(e_list)):
            this_ax.vlines(
                x=e_list[i][0],
                ymin=min(e_list[i - 1][1], e_list[i][1]),
                ymax=max(e_list[i - 1][1], e_list[i][1]),
            )
            this_ax.hlines(y=e_list[i - 1][1], xmin=e_list[i - 1][0], xmax=e_list[i][0])

    this_ax.set_xlabel("Filtration")
    this_ax.set_ylabel("Euler Characteristic")
    return this_ax


# given the list of changes to the EC and a filtration value
# returns the EC at that filtration value
def EC_at_filtration(ecc_list, f):
    """
    Compute the Euler characteristic at a given filtration value.

    Parameters
    ----------
    ecc_list : list of [float, int]
        The ECC list as [filtration, Euler characteristic] pairs.

    f : float
        The filtration value to evaluate at.

    Returns
    -------
    ec : int
        Euler characteristic at the specified filtration.
    """

    ec = ecc_list[0][1]

    for current_ec in ecc_list:
        if current_ec[0] > f:
            break
        ec = current_ec[1]

    return ec


# computes the difference between two ECC from 0 to a max filtration value
def difference_ECC(ecc1, ecc2, max_f):
    """
    Compute the L1 distance between two Euler Characteristic Curves (ECCs)
    up to a maximum filtration value.

    Parameters
    ----------
    ecc1 : list of [float, int]
        First ECC as a list of [filtration, EC].

    ecc2 : list of [float, int]
        Second ECC as a list of [filtration, EC].

    max_f : float
        The maximum filtration value to consider.

    Returns
    -------
    difference : float
        The total L1 distance between the two ECCs over [0, max_f].
    """
    # find full list of filtration points
    filtration_steps = list(
        set(([f[0] for f in ecc1] + [f[0] for f in ecc2] + [max_f]))
    )
    filtration_steps.sort()

    difference = 0

    for i in range(1, len(filtration_steps)):
        if filtration_steps[i] > max_f:
            break

        ec_1 = EC_at_filtration(ecc1, filtration_steps[i - 1])
        ec_2 = EC_at_filtration(ecc2, filtration_steps[i - 1])

        difference += abs(ec_1 - ec_2) * (filtration_steps[i] - filtration_steps[i - 1])

    return difference


# # given a distance matrix between two pointclouds
# # returns the matrix with the optimal 1-haussdorf matching and the H1 distance
# def hausdorff_1_pointclouds(C):
#     h_matrix = np.zeros(C.shape)
#
#     minInRows = np.amin(C, axis=1)
#     for i, m in enumerate(minInRows):
#         j = np.where(C[i] == np.amin(m))[0]
#         h_matrix[i, j] = 1
#
#     return h_matrix, np.sum(ot_h1 * C)
#
#
# # given a distance matrix between two pointclouds
# # returns the matrix with the optimal 1W and the 1W distance
# import ot  # ot needed to compute wasserstein distance
# def wasserstein_1_pointclouds(C):
#     ot_emd = ot.emd([], [], C)
#     ot_emd *= ot_emd.shape[0]
#
#     return ot_emd, np.sum(ot_emd * C)
