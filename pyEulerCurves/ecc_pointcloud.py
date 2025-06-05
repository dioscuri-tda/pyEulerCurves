import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from timeit import default_timer
from ._compute_local_EC_VR import compute_contributions_vertex
import gudhi as gd

################################
## Vietoris-Rips construction ##
################################


def create_local_graph(points, i, threshold, dbg=False):
    """
    Create a local graph centered at point `i` using Vietoris-Rips criterion.

    Parameters
    ----------
    points : np.ndarray
        An (N, D) array of points.
    i : int
        Index of the center vertex in the point cloud.
    threshold : float
        Distance threshold for including edges.
    dbg : bool, optional
        If True, print debug information.

    Returns
    -------
    considered_graph : list of list of tuple
        A list representing the local graph. Each sublist corresponds to a vertex
        and contains tuples of the form (neighbor_index, distance).
    """
    center_vertex = points[i]

    # enumeration needs to start from i+1 because the center is at position i
    id_neigs_of_center_vectex = [
        j
        for j, point in enumerate(points[i + 1 :], i + 1)
        if np.linalg.norm(center_vertex - point) <= threshold
    ]

    if dbg:
        print("\t {} neigbours".format(len(id_neigs_of_center_vectex)))
    # create the center graph as a list of lists
    # each list corespond to a node and contains its neighbours with distance
    # note, edges are always of the type
    # (i, j) with i<j in the ordering.
    # This is to save space, we do not need the edge (j, i)
    # In practice, we are building onlty the upper triangular part
    # of the adjecency matrix

    considered_graph = []

    # add central vertex
    mapped_center_vertex = []
    # enumeration needs to start from 1 because 0 is the center
    for j, neigh in enumerate(id_neigs_of_center_vectex, 1):
        mapped_center_vertex.append((j, np.linalg.norm(center_vertex - points[neigh])))

    considered_graph.append(mapped_center_vertex)

    # add the rest
    for j, neigh in enumerate(id_neigs_of_center_vectex, 1):

        neighbours_of_j = []

        # add the others
        # note that the index k starts from 1, be careful with the indexing
        for z, other_neigh in enumerate(id_neigs_of_center_vectex[j:], j + 1):
            dist = np.linalg.norm(points[neigh] - points[other_neigh])
            if dist <= threshold:
                neighbours_of_j.append((z, dist))

        considered_graph.append(neighbours_of_j)

    return considered_graph


def compute_contributions_single_vertex(
    point_cloud, i, epsilon, max_dimension=-1, dbg=False, measure_times=False
):
    """
    Compute Euler characteristic curve contributions for a single vertex.

    Parameters
    ----------
    point_cloud : np.ndarray
        An (N, D) array representing the point cloud.
    i : int
        Index of the vertex in the point cloud to analyze.
    epsilon : float
        Maximum distance to consider for edge inclusion in the VR complex.
    max_dimension : int, optional
        Maximum dimension of simplices to include. Default is -1 (no limit).
    dbg : bool, optional
        If True, print debug output.
    measure_times : bool, optional
        If True, measure the time taken for computation.

    Returns
    -------
    local_ECC : dict
        Dictionary with filtration values as keys and Euler characteristic contributions as values.
    number_of_simplices : int
        Total number of simplices found.
    largest_dimension : int
        The largest dimension among the simplices.
    my_time : float
        Time taken for the computation, in seconds (0 if not measured).
    """
    if dbg:
        print("point {}".format(i), end="")
    graph_i = create_local_graph(point_cloud, i, epsilon, dbg)
    if measure_times:
        start = default_timer()
        local_ECC, number_of_simplices, largest_dimension = (
            compute_contributions_vertex(graph_i, max_dimension, False)
        )
        my_time = default_timer() - start
    else:
        local_ECC, number_of_simplices, largest_dimension = (
            compute_contributions_vertex(graph_i, max_dimension, False)
        )
        my_time = 0

    return local_ECC, number_of_simplices, largest_dimension, my_time


def compute_local_contributions_VR(
    point_cloud, epsilon, max_dimension=-1, workers=1, dbg=False, measure_times=False
):
    """
    Compute local Euler characteristic contributions using the Vietoris-Rips complex.

    Parameters
    ----------
    point_cloud : np.ndarray
        An (N, D) array representing the point cloud.
    epsilon : float
        Distance threshold for VR complex construction.
    max_dimension : int, optional
        Maximum dimension of simplices to consider. Default is -1 (no limit).
    workers : int, optional
        Number of processes to use in parallel. Default is 1.
    dbg : bool, optional
        If True, print debug output.
    measure_times : bool, optional
        If True, measure the computation time for each vertex.

    Returns
    -------
    total_ECC : list of tuple
        Sorted list of (filtration_value, contribution) tuples.
    num_simplices_list : list of int
        Number of simplices for each point's local complex.
    largest_dimension_list : list of int
        Largest dimension of simplices per local complex.
    times_list : list of float
        Computation time for each local contribution.
    """
    # for each point, create its local graph and find all the
    # simplices in its star
    if dbg:
        print("compute local contributions")
        print("point cloud size {}".format(point_cloud.shape))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        ECC_list, num_simplices_list, largest_dimension_list, times_list = zip(
            *executor.map(
                compute_contributions_single_vertex,
                itertools.repeat(point_cloud),
                [i for i in range(len(point_cloud))],
                itertools.repeat(epsilon),
                itertools.repeat(max_dimension),
                itertools.repeat(dbg),
                itertools.repeat(measure_times),
            )
        )

    total_ECC = dict()

    for single_ECC in ECC_list:
        for key in single_ECC:
            total_ECC[key] = total_ECC.get(key, 0) + single_ECC[key]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECC:
        if total_ECC[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECC[key]

    return (
        sorted(list(total_ECC.items()), key=lambda x: x[0]),
        num_simplices_list,
        largest_dimension_list,
        times_list,
    )


################################
## alpha construction ##
################################


def compute_local_contributions_alpha(point_cloud, dbg=False):
    """
    Compute Euler characteristic contributions using the Alpha complex.
    Uses GUDHI to explicitely build the entire complex.

    Parameters
    ----------
    point_cloud : np.ndarray
        An (N, D) array representing the point cloud.
    dbg : bool, optional
        If True, print debug information.

    Returns
    -------
    contributions : list of tuple
        Sorted list of (filtration_value, contribution) tuples.
    num_simplices : int
        Total number of simplices in the Alpha complex.
    """
    # create an alpha complex using gudhi and computes the contributions to the EC
    if dbg:
        print("compute local contributions")
        print("point cloud size {}".format(point_cloud.shape))

    contributions = dict()
    num_simplices = 0

    alpha_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree()

    contributions = dict()

    for s, f in simplex_tree.get_filtration():
        contributions[f] = contributions.get(f, 0) + (-1) ** (len(s) - 1)
        num_simplices += 1

    # remove the contributions that are 0
    to_del = []
    for key in contributions:
        if contributions[key] == 0:
            to_del.append(key)
    for key in to_del:
        del contributions[key]

    return (sorted(list(contributions.items()), key=lambda x: x[0]), num_simplices)
