from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _unit_vector(v: np.ndarray) -> np.ndarray:
    """Return normalized vector. If vector has zero norm, return it unchanged."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# Compute the bifiltration values (b_i, b_j) for the edge (i, j)
# based on the relative positions and velocity directions.
# b measures how aligned the velocity is with the edge direction.
def projections_for_edge_2(
    i: int, j: int, positions: np.ndarray, velocities: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the bifiltration pair (b_min, b_max) for an edge between nodes i and j.

    - The edge direction vector points from i to j.
    - We compute directional alignment of velocity vectors with this edge.
    - b = 1 - dot(edge_unit, velocity_unit)
      which measures how much velocity deviates from the edge direction.

    Returns:
        (min(alpha_i, alpha_j), max(alpha_i, alpha_j))
    """
    p_i, p_j = positions[i], positions[j]
    v_i, v_j = velocities[i], velocities[j]
    edge = p_j - p_i

    # If two points coincide, assign trivial bifiltration.
    if np.allclose(edge, 0):
        return 1.0, 1.0

    # Compute unit direction vectors
    u = _unit_vector(edge)
    v_i_u = _unit_vector(v_i) if np.linalg.norm(v_i) != 0 else v_i
    v_j_u = _unit_vector(v_j) if np.linalg.norm(v_j) != 0 else v_j

    # b_i measures how aligned vertex i's velocity is with the edge
    alpha_i = 1.0 - float(np.dot(u, v_i_u))

    # b_j uses the opposite direction (edge reversed)
    alpha_j = 1.0 - float(np.dot(-u, v_j_u))

    #######################
    # zero_vel = np.sum(np.linalg.norm(velocities, axis=1)==0)
    # print("Number of zero velocities:", zero_vel)
    #######################

    return float(min(alpha_i, alpha_j)), float(max(alpha_i, alpha_j))


# For each vertex compute the minimal bifiltration obtained from
# all incident edges (within radius epsilon).
# If vertex has no neighbors → (inf, inf)
def compute_vertex_bifiltrations(
    positions: np.ndarray, velocities: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    For each vertex i:
        compute (x_v, y_v) = min over neighbors j of (x_e(i,j), y_e(i,j))

    If no neighbors exist within epsilon assign (1.0, 1.0).
    """
    n = len(positions)

    # Search neighbors within radius epsilon
    neigh = NearestNeighbors(radius=epsilon, algorithm="auto")
    neigh.fit(positions)

    # radius_neighbors returns indices of all neighbors (including itself)
    radius_results = neigh.radius_neighbors(positions, return_distance=False)

    vertex_filts = np.full((n, 2), np.inf, dtype=float)

    for i, neighs in enumerate(radius_results):
        best_x = np.inf
        best_y = np.inf

        for j in neighs:
            if j == i:
                continue

            xe, ye = projections_for_edge_2(i, j, positions, velocities)

            # Take coordinate-wise minima
            if xe < best_x:
                best_x = xe
            if ye < best_y:
                best_y = ye

        # If isolated vertex assign default filtration
        if best_x == np.inf:
            vertex_filts[i] = np.array([1.0, 1.0], dtype=float)
        else:
            vertex_filts[i] = np.array([best_x, best_y], dtype=float)

    ##############
    # num_all_isolated = np.sum(np.all(vertex_filts == np.array([1.0, 1.0]), axis=1))
    # print(f"Vertex filtrations: {num_all_isolated}/{n} are [1.0,1.0] (treated as isolated)")
    # print("Sample vertex filtrations:", vertex_filts[:10])
    ##############
    vertex_filts = vertex_filts.astype(float)

    return vertex_filts


def create_local_graph(
    positions: np.ndarray,
    velocities: np.ndarray,
    i: int,
    threshold: float,
    dbg: bool = False,
):
    center_vertex = positions[i]

    # global indices of neighbors (only j > i kept in two loops before - ale lepiej wziąć wszystkich != i)
    id_neigs_of_center_vectex = [
        j
        for j, point in enumerate(positions)
        if j != i and np.linalg.norm(center_vertex - point) <= threshold
    ]

    if dbg:
        print("neighbors (global indices):", id_neigs_of_center_vectex)

    # map global -> local index: center has local 0, neighbors 1..k
    local_idx_of_global = {i: 0}
    for local_idx, g in enumerate(id_neigs_of_center_vectex, start=1):
        local_idx_of_global[g] = local_idx

    # considered_graph: adjacency list where index = local index
    k = len(id_neigs_of_center_vectex)
    considered_graph = [[] for _ in range(k + 1)]  # 0..k

    # fill adjacency for center (local 0)
    for g in id_neigs_of_center_vectex:
        local = local_idx_of_global[g]
        xe, ye = projections_for_edge_2(i, g, positions, velocities)
        considered_graph[0].append((local, (xe, ye)))

    # fill adjacency for each neighbor (local 1..k)
    for g in id_neigs_of_center_vectex:
        local_g = local_idx_of_global[g]
        neighbours_of_g = []
        for h in id_neigs_of_center_vectex:
            if h == g:
                continue
            dist = np.linalg.norm(positions[g] - positions[h])
            if dist <= threshold:
                local_h = local_idx_of_global[h]
                xe, ye = projections_for_edge_2(g, h, positions, velocities)
                neighbours_of_g.append((local_h, (xe, ye)))
        considered_graph[local_g] = neighbours_of_g

    return considered_graph, [i] + id_neigs_of_center_vectex


# Compute Euler Characteristic Profile for a SINGLE vertex.
# We build simplices of growing dimension while tracking
# coordinate-wise maximal filtration values.


def coordwise_max(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float]:
    """Return coordinate-wise maximum of two 2D points."""
    return (max(a[0], b[0]), max(a[1], b[1]))


def compute_ECP_single_vertex(
    considered_graph: List[List[Tuple[int, Tuple[float, float]]]],
    vertex_filtrations_local: np.ndarray,
    dbg: bool = False,
):
    """
    Build local simplicial complex around a vertex and compute the
    Euler Characteristic Profile (ECP) as a function over 2D bifiltration.

    considered_graph:
        adjacency list with bifiltration on edges

    vertex_filtrations_local:
        filtrations of vertices in local numbering

    ECP:
        dictionary mapping (x,y) -> integer contribution
    """
    simplices_in_current_dimension = []  # list of simplices in the current dimension.
    filtration_of_those_simplices = (
        []
    )  # list tells what are the filtration values of the simplices in the simplices_in_current_dimension vector.
    ECC: Dict[Tuple[float, float], int] = dict()

    # Add 0-simplex (the vertex)
    v0_filtration = tuple(vertex_filtrations_local[0].tolist())
    key = (float(v0_filtration[0]), float(v0_filtration[1]))
    ECC[key] = ECC.get(key, 0) + 1

    if dbg:
        print("\n vertex filtrations : {}".format(vertex_filtrations_local))

    # Add 1-simplices (edges) from center
    for edge in considered_graph[0]:
        local_idx = edge[0]
        edge_filtration = tuple(edge[1])
        simplices_in_current_dimension.append([0, local_idx])
        filtration_of_those_simplices.append(edge_filtration)

        if dbg:
            print(f"######Change of ECC at level {edge_filtration} by -1")

        # Edges contribute -1
        key = (float(edge_filtration[0]), float(edge_filtration[1]))
        ECC[key] = ECC.get(key, 0) - 1

    if dbg:
        print("simplices_in_current_dimension :")
        for i, s in enumerate(simplices_in_current_dimension):
            print("[{}] --> {}".format(s, filtration_of_those_simplices[i]))

    # Determine common neighbors (for higher simplices)
    common_neighs = []
    number_of_simplices = 1 + len(simplices_in_current_dimension)

    for simplex in simplices_in_current_dimension:
        the_other_vertex = simplex[1]
        neighs = [nbr for nbr, _ in considered_graph[the_other_vertex]]
        common_neighs.append(neighs)

    if dbg:
        print("common_neighs:", common_neighs)

    # Precompute neighbors of each vertex (as sets)
    neighs_of_vertices = [
        set([v for v, _ in vertex_list]) for vertex_list in considered_graph
    ]

    dimm = 1
    dimension = 2  # move to triangles

    # Iteratively build higher simplices
    while len(simplices_in_current_dimension) > 0:

        new_simplices_in_current_dimension = []
        new_filtration_of_those_simplices = []
        new_common_neighs = []

        # Loop over parent simplices
        for i in range(len(simplices_in_current_dimension)):

            parent_simplex = simplices_in_current_dimension[i]
            parent_filtration = filtration_of_those_simplices[i]

            # Try adding each common neighbor
            for j in range(len(common_neighs[i])):

                candidate = common_neighs[i][j]

                new_simplex = parent_simplex.copy()
                new_simplex.append(candidate)
                number_of_simplices += 1

                if dbg:
                    print("Adding new simplex : {}".format(new_simplex))

                # Compute filtration = max over parent + new edges
                filtration_of_this_simplex = parent_filtration

                # Check edges connecting new vertex to all vertices in parent_simplex
                for vertex_local in parent_simplex:
                    # Search adjacency list of vertex_local or candidate
                    found = False

                    for nbr_local, edge_bif in considered_graph[vertex_local]:
                        if nbr_local == candidate:
                            filtration_of_this_simplex = coordwise_max(
                                filtration_of_this_simplex, edge_bif
                            )
                            found = True
                            break

                    if not found:
                        for nbr_local, edge_bif in considered_graph[candidate]:
                            if nbr_local == vertex_local:
                                filtration_of_this_simplex = coordwise_max(
                                    filtration_of_this_simplex, edge_bif
                                )
                                found = True
                                break

                filtration_of_this_simplex = tuple(filtration_of_this_simplex)

                new_filtration_of_those_simplices.append(filtration_of_this_simplex)

                if dbg:
                    print(
                        f"#####Change of ECC at level {filtration_of_this_simplex} by {dimm}"
                    )

                a, b = filtration_of_this_simplex
                key = (float(a), float(b))
                ECC[key] = ECC.get(key, 0) + dimm

                # Determine new common neighbors for this extended simplex
                neighs_of_new_simplex = []
                new_vertex = candidate

                for k in range(len(common_neighs[i])):
                    if common_neighs[i][k] in neighs_of_vertices[new_vertex]:
                        neighs_of_new_simplex.append(common_neighs[i][k])

                new_common_neighs.append(neighs_of_new_simplex)

                new_simplices_in_current_dimension.append(new_simplex)

        simplices_in_current_dimension = new_simplices_in_current_dimension
        filtration_of_those_simplices = new_filtration_of_those_simplices
        common_neighs = new_common_neighs

        dimension += 1
        dimm = dimm * (-1)  # change sign for next dimension

    return ECC, number_of_simplices


def compute_local_contributions(
    positions: np.ndarray, velocities: np.ndarray, epsilon: float, dbg: bool = False
):
    """
    Compute local bifiltration-based contributions for each vertex.
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n, d) containing point coordinates.
    velocities : np.ndarray
        Array of shape (n, d) containing velocity vectors associated with the points.
    epsilon : float
        Neighborhood radius used to construct local graphs.
    dbg : bool, optional
        If True, enables debug output.

    Returns
    -------
    contributions : List[Tuple[Tuple[float, float], int]]
        A sorted list of (bifiltration_value, contribution) pairs.
        Bifiltration values are 2D tuples.
    total_number_of_simplices : int
        Total number of simplices processed across all local computations.
    """

    n = len(positions)

    # Compute global vertex bifiltrations
    vertex_filts_global = compute_vertex_bifiltrations(positions, velocities, epsilon)

    ############
    # vf = vertex_filts_global
    # unique_rows, counts = np.unique(vf, axis=0, return_counts=True)
    # print("Unique vertex filtrations and counts:")
    # for row, c in zip(unique_rows, counts):
    #    print(row, c)
    # print("Sample vertex filtrations:", vf[:20])

    ############

    ECP_list = []
    total_number_of_simplices = 0

    for idx in tqdm(range(n)):
        graph_i, global_ids = create_local_graph(
            positions, velocities, idx, epsilon, dbg=dbg
        )

        # Extract local vertex bifiltrations following the order in global_ids
        vertex_filtrations_local = np.array(
            [vertex_filts_global[g] for g in global_ids]
        )

        local_ECP, n_simp = compute_ECP_single_vertex(
            graph_i, vertex_filtrations_local, dbg=dbg
        )

        ECP_list.append(local_ECP)
        total_number_of_simplices += n_simp

    # Aggregate all local ECP contributions
    total_ECP: Dict[Tuple[float, float], int] = dict()
    for local_map in ECP_list:
        for key, val in local_map.items():
            total_ECP[key] = total_ECP.get(key, 0) + val

    # Remove zero contributions
    to_del = []
    for key in total_ECP:
        if total_ECP[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECP[key]

    # Sort by the 2D bifiltration key
    sorted_contributions = sorted(
        [((float(k[0]), float(k[1])), int(v)) for k, v in total_ECP.items()],
        key=lambda x: x[0],
    )

    return sorted_contributions, total_number_of_simplices
