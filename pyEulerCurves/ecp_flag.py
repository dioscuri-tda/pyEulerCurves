from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral, Real
from typing import Literal

from sklearn.base import BaseEstimator, TransformerMixin

FiltrationValue = tuple[float, ...]
ECPContribution = tuple[FiltrationValue, int]
Traversal = Literal["dfs", "bfs"]


class FilteredGraph:
    """
    Sparse representation of a filtered graph for flag-complex ECP computations.

    Attributes
    ----------
    vertex_filtrations : tuple[FiltrationValue, ...]
        Normalized vertex filtration values.
    edges : tuple[tuple[int, int], ...]
        Normalized undirected edges with smaller endpoint first.
    edge_filtrations : tuple[FiltrationValue, ...]
        Normalized edge filtration values aligned with ``edges``.
    filtration_dimension : int
        Number of coordinates in each filtration value.
    adjacency : tuple[frozenset[int], ...]
        Sparse adjacency sets for each vertex.
    num_vertices : int
        Number of vertices.
    num_edges : int
        Number of edges.
    """

    def __init__(
        self,
        vertex_filtrations: Sequence[float | Sequence[float]],
        edges: Sequence[tuple[int, int]],
        edge_filtrations: Sequence[float | Sequence[float]],
    ) -> None:
        self.vertex_filtrations = tuple(
            self._normalize_filtration_value(value, f"vertex_filtrations[{index}]")
            for index, value in enumerate(vertex_filtrations)
        )

        if len(edges) != len(edge_filtrations):
            raise ValueError("edges and edge_filtrations must have the same length.")

        self.edges = self._normalize_edges(edges, len(self.vertex_filtrations))
        self.edge_filtrations = tuple(
            self._normalize_filtration_value(value, f"edge_filtrations[{index}]")
            for index, value in enumerate(edge_filtrations)
        )

        self.filtration_dimension = self._validate_filtration_dimensions(
            self.vertex_filtrations,
            self.edge_filtrations,
        )
        self._validate_filtration(
            self.vertex_filtrations,
            self.edges,
            self.edge_filtrations,
        )

        adjacency = [set() for _ in self.vertex_filtrations]
        edge_filtration_map: list[dict[int, FiltrationValue]] = [
            {} for _ in self.vertex_filtrations
        ]
        for (source, target), filtration in zip(
            self.edges,
            self.edge_filtrations,
            strict=True,
        ):
            adjacency[source].add(target)
            adjacency[target].add(source)
            edge_filtration_map[source][target] = filtration
            edge_filtration_map[target][source] = filtration

        self.adjacency = tuple(frozenset(neighbors) for neighbors in adjacency)
        self._edge_filtration_map = tuple(edge_filtration_map)

    @classmethod
    def from_graph_data(
        cls,
        vertex_filtrations: Sequence[float | Sequence[float]],
        edges: Sequence[tuple[int, int]],
        edge_filtrations: Sequence[float | Sequence[float]],
    ) -> FilteredGraph:
        """Construct a filtered graph from complete vertex and edge data."""
        return cls(
            vertex_filtrations=vertex_filtrations,
            edges=edges,
            edge_filtrations=edge_filtrations,
        )

    @classmethod
    def from_networkx(
        cls,
        graph: object,
        vertex_attr: str = "filtration",
        edge_attr: str = "filtration",
    ) -> FilteredGraph:
        """
        Construct a filtered graph from a NetworkX graph.

        NetworkX is optional. This method imports it lazily and raises an
        informative error if NetworkX is not available.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX is required to convert a NetworkX graph. "
                "Install networkx or pass a FilteredGraph instance."
            ) from exc

        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX graph.")
        if graph.is_directed():
            raise ValueError("FilteredGraph only supports undirected graphs.")

        nodes = list(graph.nodes)
        node_to_index = {node: index for index, node in enumerate(nodes)}

        try:
            vertex_filtrations = [graph.nodes[node][vertex_attr] for node in nodes]
        except KeyError as exc:
            raise ValueError(
                f"All NetworkX nodes must define the {vertex_attr!r} attribute."
            ) from exc

        edges: list[tuple[int, int]] = []
        edge_filtrations = []
        try:
            edge_iter = graph.edges(data=True)
        except TypeError as exc:
            raise TypeError("graph must provide NetworkX-style edges(data=True).") from exc

        for source, target, data in edge_iter:
            edges.append((node_to_index[source], node_to_index[target]))
            try:
                edge_filtrations.append(data[edge_attr])
            except KeyError as exc:
                raise ValueError(
                    f"All NetworkX edges must define the {edge_attr!r} attribute."
                ) from exc

        return cls(
            vertex_filtrations=vertex_filtrations,
            edges=edges,
            edge_filtrations=edge_filtrations,
        )

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the graph."""
        return len(self.vertex_filtrations)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    def vertex_filtration(self, vertex: int) -> FiltrationValue:
        """Return the filtration value of a vertex."""
        try:
            return self.vertex_filtrations[vertex]
        except IndexError as exc:
            raise ValueError(f"Vertex {vertex} is not present.") from exc

    def edge_filtration(self, source: int, target: int) -> FiltrationValue:
        """Return the filtration value of an edge."""
        try:
            return self._edge_filtration_map[source][target]
        except (IndexError, KeyError) as exc:
            raise ValueError(f"Edge ({source}, {target}) is not present.") from exc

    def filtration_values(self) -> tuple[FiltrationValue, ...]:
        """Return all vertex and edge filtration values."""
        return self.vertex_filtrations + self.edge_filtrations

    @staticmethod
    def _normalize_filtration_value(
        value: float | Sequence[float],
        name: str,
    ) -> FiltrationValue:
        """Convert a scalar or coordinate sequence to a filtration tuple."""
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be a real number or a sequence of real numbers."
            )
        if isinstance(value, Real):
            return (float(value),)
        if isinstance(value, (str, bytes)):
            raise ValueError(
                f"{name} must be a real number or a sequence of real numbers."
            )

        try:
            coordinates = tuple(float(coordinate) for coordinate in value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a real number or a sequence of real numbers."
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-numeric coordinate.") from exc

        if len(coordinates) == 0:
            raise ValueError(f"{name} must contain at least one coordinate.")
        return coordinates

    @staticmethod
    def _normalize_edges(
        edges: Sequence[tuple[int, int]],
        num_vertices: int,
    ) -> tuple[tuple[int, int], ...]:
        """Validate edges and store each undirected edge in canonical order."""
        seen: set[tuple[int, int]] = set()
        normalized_edges = []
        for index, edge in enumerate(edges):
            try:
                source, target = edge
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"edges[{index}] must contain exactly two endpoints."
                ) from exc

            if (
                isinstance(source, bool)
                or isinstance(target, bool)
                or not isinstance(source, Integral)
                or not isinstance(target, Integral)
            ):
                raise ValueError(f"edges[{index}] endpoints must be integers.")

            source = int(source)
            target = int(target)
            if source == target:
                raise ValueError("FilteredGraph does not support self-loops.")
            if (
                source < 0
                or source >= num_vertices
                or target < 0
                or target >= num_vertices
            ):
                raise ValueError(f"edges[{index}] contains an out-of-range endpoint.")

            edge_key = (source, target) if source < target else (target, source)
            if edge_key in seen:
                raise ValueError("FilteredGraph does not support duplicate edges.")
            seen.add(edge_key)
            normalized_edges.append(edge_key)

        return tuple(normalized_edges)

    @staticmethod
    def _validate_filtration_dimensions(
        vertex_filtrations: Sequence[FiltrationValue],
        edge_filtrations: Sequence[FiltrationValue],
    ) -> int:
        """Return the common filtration dimension after validating consistency."""
        values = tuple(vertex_filtrations) + tuple(edge_filtrations)
        if not values:
            return 0

        dimension = len(values[0])
        for value in values:
            if len(value) != dimension:
                raise ValueError("All filtration values must have the same dimension.")
        return dimension

    @staticmethod
    def _validate_filtration(
        vertex_filtrations: Sequence[FiltrationValue],
        edges: Sequence[tuple[int, int]],
        edge_filtrations: Sequence[FiltrationValue],
    ) -> None:
        """Validate coordinatewise compatibility of edge and vertex filtrations."""
        for (source, target), edge_filtration in zip(
            edges,
            edge_filtrations,
            strict=True,
        ):
            if not FilteredGraph._coordinatewise_leq(
                vertex_filtrations[source],
                edge_filtration,
            ):
                raise ValueError(
                    "Edge filtration values must be coordinatewise greater than "
                    "or equal to endpoint vertex filtrations."
                )
            if not FilteredGraph._coordinatewise_leq(
                vertex_filtrations[target],
                edge_filtration,
            ):
                raise ValueError(
                    "Edge filtration values must be coordinatewise greater than "
                    "or equal to endpoint vertex filtrations."
                )

    @staticmethod
    def _coordinatewise_leq(
        left: FiltrationValue,
        right: FiltrationValue,
    ) -> bool:
        """Return whether left is coordinatewise less than or equal to right."""
        return all(
            left_value <= right_value
            for left_value, right_value in zip(left, right, strict=True)
        )


_WORKER_GRAPH: FilteredGraph | None = None
_WORKER_FORWARD_NEIGHBORS: Sequence[set[int]] | None = None
_WORKER_TRAVERSAL: Traversal | None = None


class ECP_from_filtered_graph(TransformerMixin, BaseEstimator):
    """
    Compute the Euler Characteristic Profile of a filtered graph's flag complex.

    Transformer following the scikit-learn API. The input is a single filtered graph,
    and ``transform`` returns the ECP of its filtered flag complex.

    Parameters
    ----------
    workers : int, default=1
        Number of worker processes. ``1`` runs sequentially and ``-1`` uses all
        available CPUs.
    traversal : {"dfs", "bfs"}, default="dfs"
        Traversal strategy for the local clique-enumeration tree used to
        enumerate simplices of the flag complex. ``"dfs"`` uses depth-first
        search and ``"bfs"`` uses breadth-first search.
    use_degeneracy_ordering : bool, default=False
        Whether to orient the clique enumeration along a degeneracy ordering of
        the vertices (computed in linear time) instead of the natural vertex
        order. This bounds each vertex's forward-neighbour set by the graph degeneracy,
        but adds an O(V+E) overhead, so it is off by default.

    Attributes
    ----------
    num_simplices : int
        Total number of simplices in the flag complex (set by ``transform``).
    largest_dimension : int
        Largest simplex dimension present, or ``-1`` if there are none.
    dim_counts : dict of int to int
        Mapping from dimension to the number of simplices of that dimension.
    """

    def __init__(self, workers=1, traversal="dfs", use_degeneracy_ordering=False):
        self.workers = workers
        self.traversal = traversal
        self.use_degeneracy_ordering = use_degeneracy_ordering

    def fit(self, X, y=None):
        """No-op; present for scikit-learn API compatibility. Returns ``self``."""
        return self

    def transform(self, X):
        """
        Compute the ECP of the flag complex of ``X``.

        Parameters
        ----------
        X : pyEulerCurves.FilteredGraph or networkx.Graph
            Filtered graph whose vertices and edges carry scalar or vector
            filtration values. If a NetworkX graph is passed, node and edge
            filtrations are read from the ``"filtration"`` attribute.

        Returns
        -------
        list of (filtration_value, contribution) pairs, sorted by filtration_value
            The Euler Characteristic Profile of the flag complex. Each
            contribution is an integer representing the net contribution of
            simplices with the given filtration value to the Euler
            characteristic.
        """
        ecp, stats = _compute_ecp(
            X, self.workers, self.traversal, self.use_degeneracy_ordering
        )
        self.num_simplices = stats["num_simplices"]
        self.largest_dimension = stats["largest_dimension"]
        self.dim_counts = stats["dim_counts"]
        return ecp


def _compute_ecp(
    graph: object,
    workers: int,
    traversal: Traversal,
    use_degeneracy_ordering: bool = False,
) -> tuple[list[ECPContribution], dict[str, object]]:
    """Compute the ECP of ``graph`` and accompanying simplex-count statistics."""
    filtered_graph = _normalize_graph_input(graph)
    worker_count = _normalize_workers(workers, filtered_graph.num_vertices)
    if traversal not in {"dfs", "bfs"}:
        raise ValueError("traversal must be either 'dfs' or 'bfs'.")

    if use_degeneracy_ordering:
        vertex_order = _degeneracy_ordering(filtered_graph)
    else:
        vertex_order = list(range(filtered_graph.num_vertices))
    forward_neighbors = _build_forward_adjacency(filtered_graph, vertex_order)

    totals, dim_counts = _run_vertex_jobs(
        vertex_order,
        forward_neighbors,
        filtered_graph,
        traversal,
        worker_count,
    )

    ecp = sorted(
        (filtration_value, contribution)
        for filtration_value, contribution in totals.items()
        if contribution != 0
    )

    present_dims = [dim for dim, count in dim_counts.items() if count > 0]
    stats: dict[str, object] = {
        "num_simplices": sum(dim_counts.values()),
        "largest_dimension": max(present_dims) if present_dims else -1,
        "dim_counts": dict(sorted(dim_counts.items())),
    }

    return ecp, stats


def _normalize_graph_input(graph: object) -> FilteredGraph:
    """Convert supported graph inputs to a FilteredGraph."""
    if isinstance(graph, FilteredGraph):
        return graph

    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is not None and isinstance(graph, nx.Graph):
        return FilteredGraph.from_networkx(graph)

    raise TypeError("graph must be a FilteredGraph or a NetworkX graph.")


def _normalize_workers(workers: int, num_vertices: int) -> int:
    """Validate and cap the requested worker count."""
    if not isinstance(workers, Integral) or isinstance(workers, bool):
        raise ValueError("workers must be an integer.")
    workers = int(workers)
    if workers == -1:
        workers = os.cpu_count() or 1
    elif workers == 0 or workers < -1:
        raise ValueError("workers must be a positive integer or -1.")

    if num_vertices == 0:
        return 1
    return min(workers, num_vertices)


def _degeneracy_ordering(graph: FilteredGraph) -> list[int]:
    """Return the degeneracy (min-degree peeling) order in O(V + E).

    This is the order in which vertices are removed when repeatedly deleting a
    vertex of minimum remaining degree. It is computed with the Batagelj-Zaversnik
    bin-sort algorithm. Ties in current degree are broken by ascending vertex index,
    matching the previous order.
    """
    num_vertices = graph.num_vertices
    if num_vertices == 0:
        return []

    adjacency = graph.adjacency
    degree = [len(adjacency[vertex]) for vertex in range(num_vertices)]
    max_degree = max(degree)

    # bin_start[d] is the position in ``ordered_by_degree`` where the block of
    # vertices with current degree d begins.
    bin_start = [0] * (max_degree + 2)
    for vertex_degree in degree:
        bin_start[vertex_degree + 1] += 1
    for d in range(1, max_degree + 2):
        bin_start[d] += bin_start[d - 1]

    # Bin-sort vertices by degree (ascending vertex index within each degree).
    ordered_by_degree = [0] * num_vertices
    position = [0] * num_vertices
    cursor = bin_start[:]
    for vertex in range(num_vertices):
        slot = cursor[degree[vertex]]
        ordered_by_degree[slot] = vertex
        position[vertex] = slot
        cursor[degree[vertex]] += 1

    order = []
    for index in range(num_vertices):
        vertex = ordered_by_degree[index]
        order.append(vertex)
        # Peel ``vertex``: decrement the degree of each not-yet-removed
        # neighbor by moving it to the front of its degree bin.
        for neighbor in adjacency[vertex]:
            if degree[neighbor] > degree[vertex]:
                neighbor_degree = degree[neighbor]
                neighbor_pos = position[neighbor]
                swap_pos = bin_start[neighbor_degree]
                swap_vertex = ordered_by_degree[swap_pos]
                if neighbor != swap_vertex:
                    position[neighbor] = swap_pos
                    position[swap_vertex] = neighbor_pos
                    ordered_by_degree[neighbor_pos] = swap_vertex
                    ordered_by_degree[swap_pos] = neighbor
                bin_start[neighbor_degree] += 1
                degree[neighbor] -= 1

    return order


def _build_forward_adjacency(
    graph: FilteredGraph,
    vertex_order: Sequence[int],
) -> list[set[int]]:
    rank = {vertex: index for index, vertex in enumerate(vertex_order)}
    forward_neighbors = [set() for _ in range(graph.num_vertices)]
    for source, target in graph.edges:
        if rank[source] < rank[target]:
            forward_neighbors[source].add(target)
        else:
            forward_neighbors[target].add(source)
    return forward_neighbors


def _run_vertex_jobs(
    vertex_order: Sequence[int],
    forward_neighbors: Sequence[set[int]],
    graph: FilteredGraph,
    traversal: Traversal,
    workers: int,
) -> tuple[dict[FiltrationValue, int], dict[int, int]]:
    """Compute and merge local ECP contributions across vertex chunks."""
    if workers == 1:
        chunk_results = [
            _compute_chunk(
                vertex_order,
                forward_neighbors,
                graph,
                traversal,
            )
        ]
    else:
        chunks = _chunk_vertices(vertex_order, workers)
        chunk_results = _map_chunks_to_workers(
            chunks,
            graph,
            forward_neighbors,
            traversal,
            workers,
        )

    totals: dict[FiltrationValue, int] = defaultdict(int)
    dim_counts: dict[int, int] = defaultdict(int)
    for chunk_totals, chunk_dim_counts in chunk_results:
        for filtration_value, contribution in chunk_totals.items():
            totals[filtration_value] += contribution
        for dim, count in chunk_dim_counts.items():
            dim_counts[dim] += count

    return dict(totals), dict(dim_counts)


def _chunk_vertices(vertex_order: Sequence[int], workers: int) -> list[list[int]]:
    """Split ordered vertices into chunks for process workers.

    Vertices are dealt out in strided (round-robin) fashion rather than as
    contiguous slices. The per-vertex cost can vary systematically with a
    vertex's position in the chosen order (e.g. under a degeneracy ordering
    early vertices have many forward neighbors and later ones have few), so
    contiguous slices can be badly unbalanced. Striding spreads heavy and light
    vertices more evenly across chunks regardless of the ordering.
    """
    vertices = list(vertex_order)
    if not vertices:
        return []

    chunk_count = min(len(vertices), max(1, workers * 4))
    return [vertices[start::chunk_count] for start in range(chunk_count)]


def _map_chunks_to_workers(
    chunks: Sequence[Sequence[int]],
    graph: FilteredGraph,
    forward_neighbors: Sequence[set[int]],
    traversal: Traversal,
    workers: int,
) -> list[tuple[dict[FiltrationValue, int], dict[int, int]]]:
    """Compute vertex chunks in worker processes.

    The graph and forward-adjacency are large and read-only. Where ``fork`` is
    available (Linux, macOS), workers inherit them via copy-on-write by reading
    module globals set before the pool is created, which avoids pickling the
    graph to every worker -- the dominant startup cost on large inputs. On
    platforms without ``fork`` (e.g. Windows), fall back to shipping the data
    through the pool initializer.
    """
    if "fork" in mp.get_all_start_methods():
        global _WORKER_GRAPH, _WORKER_FORWARD_NEIGHBORS, _WORKER_TRAVERSAL
        _WORKER_GRAPH = graph
        _WORKER_FORWARD_NEIGHBORS = forward_neighbors
        _WORKER_TRAVERSAL = traversal
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("fork"),
            ) as executor:
                return list(executor.map(_compute_initialized_chunk, chunks))
        finally:
            _WORKER_GRAPH = None
            _WORKER_FORWARD_NEIGHBORS = None
            _WORKER_TRAVERSAL = None

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(graph, forward_neighbors, traversal),
    ) as executor:
        return list(executor.map(_compute_initialized_chunk, chunks))


def _initialize_worker(
    graph: FilteredGraph,
    forward_neighbors: Sequence[set[int]],
    traversal: Traversal,
) -> None:
    """Store read-only computation context in a worker process."""
    global _WORKER_GRAPH
    global _WORKER_FORWARD_NEIGHBORS
    global _WORKER_TRAVERSAL

    _WORKER_GRAPH = graph
    _WORKER_FORWARD_NEIGHBORS = forward_neighbors
    _WORKER_TRAVERSAL = traversal


def _compute_initialized_chunk(
    vertices: Sequence[int],
) -> tuple[dict[FiltrationValue, int], dict[int, int]]:
    """Compute a vertex chunk using worker-initialized graph context."""
    if (
        _WORKER_GRAPH is None
        or _WORKER_FORWARD_NEIGHBORS is None
        or _WORKER_TRAVERSAL is None
    ):
        raise RuntimeError("Worker process was not initialized.")

    return _compute_chunk(
        vertices,
        _WORKER_FORWARD_NEIGHBORS,
        _WORKER_GRAPH,
        _WORKER_TRAVERSAL,
    )


def _compute_chunk(
    vertices: Sequence[int],
    forward_neighbors: Sequence[set[int]],
    graph: FilteredGraph,
    traversal: Traversal,
) -> tuple[dict[FiltrationValue, int], dict[int, int]]:
    """Compute local ECP contributions for a vertex chunk."""
    totals: dict[FiltrationValue, int] = defaultdict(int)
    dim_counts: dict[int, int] = defaultdict(int)
    local_fn = (
        _local_ecp_contribution_dfs        
        if traversal == "dfs"
        else _local_ecp_contribution_bfs
    )

    for vertex in vertices:
        local_totals, local_dim_counts = local_fn(
            vertex,
            forward_neighbors,
            graph,
        )
        for filtration_value, contribution in local_totals.items():
            totals[filtration_value] += contribution
        for dim, count in local_dim_counts.items():
            dim_counts[dim] += count

    return dict(totals), dict(dim_counts)


def _local_ecp_contribution_bfs(
    vertex: int,
    forward_neighbors: Sequence[set[int]],
    graph: FilteredGraph,
) -> tuple[dict[FiltrationValue, int], dict[int, int]]:
    """Enumerate local flag-complex simplices using breadth-first search."""
    ecp: dict[FiltrationValue, int] = defaultdict(int)
    dim_counts: dict[int, int] = defaultdict(int)

    vertex_filtration = graph.vertex_filtrations[vertex]
    _add_ecp_contribution(ecp, vertex_filtration, 1)
    dim_counts[0] += 1

    current_simplices: list[tuple[tuple[int, ...], FiltrationValue]] = []
    for neighbor in forward_neighbors[vertex]:
        edge_filtration = graph._edge_filtration_map[vertex][neighbor]
        _add_ecp_contribution(ecp, edge_filtration, -1)
        dim_counts[1] += 1
        current_simplices.append(((vertex, neighbor), edge_filtration))

    simplex_dim = 1
    while current_simplices:
        next_dim = simplex_dim + 1
        sign = (-1) ** next_dim
        next_simplices: list[tuple[tuple[int, ...], FiltrationValue]] = []
        for simplex, filtration_value in current_simplices:
            for candidate in _common_forward_neighbors(simplex, forward_neighbors):
                new_filtration_value = _simplex_extension_filtration(
                    graph,
                    simplex,
                    candidate,
                    filtration_value,
                )
                _add_ecp_contribution(ecp, new_filtration_value, sign)
                dim_counts[next_dim] += 1
                next_simplices.append(((*simplex, candidate), new_filtration_value))
        current_simplices = next_simplices
        simplex_dim = next_dim

    return dict(ecp), dict(dim_counts)


def _local_ecp_contribution_dfs(
    vertex: int,
    forward_neighbors: Sequence[set[int]],
    graph: FilteredGraph,
) -> tuple[dict[FiltrationValue, int], dict[int, int]]:
    """Enumerate local flag-complex simplices using depth-first search."""
    ecp: dict[FiltrationValue, int] = defaultdict(int)
    dim_counts: dict[int, int] = defaultdict(int)

    vertex_filtration = graph.vertex_filtrations[vertex]
    _add_ecp_contribution(ecp, vertex_filtration, 1)
    dim_counts[0] += 1

    def extend(
        simplex: tuple[int, ...],
        filtration_value: FiltrationValue,
        simplex_dim: int,
    ) -> None:
        next_dim = simplex_dim + 1
        sign = (-1) ** next_dim
        for candidate in _common_forward_neighbors(simplex, forward_neighbors):
            new_filtration_value = _simplex_extension_filtration(
                graph,
                simplex,
                candidate,
                filtration_value,
            )
            _add_ecp_contribution(ecp, new_filtration_value, sign)
            dim_counts[next_dim] += 1
            extend((*simplex, candidate), new_filtration_value, next_dim)

    for neighbor in forward_neighbors[vertex]:
        edge_filtration = graph._edge_filtration_map[vertex][neighbor]
        _add_ecp_contribution(ecp, edge_filtration, -1)
        dim_counts[1] += 1
        extend((vertex, neighbor), edge_filtration, 1)

    return dict(ecp), dict(dim_counts)


def _common_forward_neighbors(
    simplex: tuple[int, ...],
    forward_neighbors: Sequence[set[int]],
) -> set[int]:
    """Return candidates that extend a simplex in the forward orientation."""
    return set.intersection(*(forward_neighbors[vertex] for vertex in simplex))


def _simplex_extension_filtration(
    graph: FilteredGraph,
    simplex: tuple[int, ...],
    candidate: int,
    filtration_value: FiltrationValue,
) -> FiltrationValue:
    """Return the filtration value after adding a candidate to a simplex."""
    new_filtration_value = filtration_value
    for vertex in simplex:
        edge_filtration = graph._edge_filtration_map[vertex][candidate]
        new_filtration_value = _componentwise_max(
            new_filtration_value,
            edge_filtration,
        )
    return new_filtration_value


def _add_ecp_contribution(
    ecp: dict[FiltrationValue, int],
    filtration_value: FiltrationValue,
    contribution: int,
) -> None:
    """Add a signed simplex contribution to an ECP accumulator."""
    ecp[filtration_value] += contribution


def _componentwise_max(left: FiltrationValue, right: FiltrationValue) -> FiltrationValue:
    """Return the coordinatewise maximum of two filtration values."""
    return tuple(
        max(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


__all__ = [
    "ECP_from_filtered_graph",
    "FilteredGraph",
]
