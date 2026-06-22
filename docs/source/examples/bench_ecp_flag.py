"""When does parallelizing ECP_from_filtered_graph pay off?

Sweeps over (number of vertices, Erdos-Renyi edge probability) and measures the
speedup of ``workers=-1`` over ``workers=1``. It produces two figures:

  * ``ecp_flag_sweep.png`` -- a heatmap of speedup over the (n, p) plane, and
  * ``ecp_flag_decision.png`` -- a logistic fit of P(parallel faster) against the
    sequential runtime, with the 50% break-even time and its confidence interval.

The decision figure is the practical one: it collapses the whole (n, p) plane
onto a single predictor -- how long the sequential run takes.

Run it (from the repository root)::

    uv run python docs/source/examples/bench_ecp_flag.py

Custom grid::

    uv run python docs/source/examples/bench_ecp_flag.py \
        --sizes 500 1000 2000 --probs 0.004 0.008 0.016 --seeds 0 1 2

Outputs are written to the sibling ``data/`` directory by default.
``--from-cache`` regenerates the figures from the saved measurements without
recomputing.

Note: in an ER graph the average degree is ``p * (n - 1)``, so the heavy corner
is large-n *and* large-p; ``--heavy-degree`` keeps the runtime bounded there.
"""

from __future__ import annotations

import argparse
import os
import time
from statistics import NormalDist

import numpy as np

from pyEulerCurves.ecp_flag import ECP_from_filtered_graph, FilteredGraph

# Figures and the measurement cache are written to the sibling ``data/``
# directory by default (the same place the example notebook reads them from).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")


def _cpu_label() -> str:
    """Human-readable description of the parallelism available on this machine.

    ``workers=-1`` uses ``os.cpu_count()`` (logical processors = threads). With
    hyper-threading the physical core count is the better guide to the speedup
    ceiling, so report both when they differ.
    """
    logical = os.cpu_count() or 1
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    if physical and physical != logical:
        return f"{physical} cores / {logical} threads"
    return f"{logical} threads"


def make_random_filtered_graph(
    num_vertices: int,
    avg_degree: float,
    seed: int,
) -> FilteredGraph:
    """Build an Erdos-Renyi filtered graph with a target average degree.

    Vertices get random scalar filtration values; each edge gets the max of its
    endpoints plus a nonnegative offset, so the coordinatewise monotonicity
    required by FilteredGraph holds by construction.
    """
    rng = np.random.default_rng(seed)
    # G(n, p): expected degree (n - 1) * p  ->  p = avg_degree / (n - 1)
    prob = min(1.0, avg_degree / max(1, num_vertices - 1))

    vertex_filtrations = rng.random(num_vertices)

    # Sample the strict upper triangle without materializing the full n x n mask.
    iu, ju = np.triu_indices(num_vertices, k=1)
    keep = rng.random(iu.shape[0]) < prob
    sources = iu[keep]
    targets = ju[keep]

    edges = list(zip(sources.tolist(), targets.tolist(), strict=True))
    endpoint_max = np.maximum(
        vertex_filtrations[sources],
        vertex_filtrations[targets],
    )
    offsets = rng.random(sources.shape[0])
    edge_filtrations = (endpoint_max + offsets).tolist()

    return FilteredGraph.from_graph_data(
        vertex_filtrations=vertex_filtrations.tolist(),
        edges=edges,
        edge_filtrations=edge_filtrations,
    )


def time_run(graph, workers: int, repeats: int) -> float:
    """Best wall-clock time over ``repeats`` runs."""
    best = float("inf")
    transformer = ECP_from_filtered_graph(workers=workers)
    for _ in range(repeats):
        start = time.perf_counter()
        transformer.fit_transform(graph)
        best = min(best, time.perf_counter() - start)
    return best


def run_sweep(
    sizes: list[int],
    probs: list[float],
    repeats: int,
    seeds: list[int],
    max_degree: float,
    heavy_degree: float,
) -> dict:
    cpu_label = _cpu_label()
    speedup = np.full((len(probs), len(sizes)), np.nan)
    seq_grid = np.full((len(probs), len(sizes)), np.nan)
    par_grid = np.full((len(probs), len(sizes)), np.nan)
    points: list[dict] = []  # one record per (cell, seed) for the decision plot

    print(f"parallelism: {cpu_label}, repeats={repeats}, seeds={seeds}")
    print(f"skipping cells with avg degree > {max_degree} (blow-up guard)\n")
    print(
        f"{'n':>7} {'p':>8} {'avg_deg':>8} {'edges':>9} "
        f"{'seq (s)':>9} {'par (s)':>9} {'speedup':>8}"
    )
    for i, p in enumerate(probs):
        for j, n in enumerate(sizes):
            avg_degree = p * (n - 1)
            if avg_degree > max_degree:
                print(
                    f"{n:>7} {p:>8.4f} {avg_degree:>8.1f} {'-':>9} "
                    f"{'skipped (too dense)':>30}",
                    flush=True,
                )
                continue
            # Heavy cells are very slow (the dense corner has millions of
            # simplices), so measure those once instead of once per seed.
            cell_seeds = seeds if avg_degree <= heavy_degree else seeds[:1]
            cell_speedups = []
            cell_seq = []
            cell_par = []
            for seed in cell_seeds:
                graph = make_random_filtered_graph(n, avg_degree, seed)
                seq_t = time_run(graph, workers=1, repeats=repeats)
                par_t = time_run(graph, workers=-1, repeats=repeats)
                su = seq_t / par_t if par_t > 0 else float("nan")
                cell_speedups.append(su)
                cell_seq.append(seq_t)
                cell_par.append(par_t)
                points.append(
                    {
                        "n": n,
                        "p": p,
                        "avg_degree": avg_degree,
                        "seed": seed,
                        "seq": seq_t,
                        "par": par_t,
                        "speedup": su,
                    }
                )
            # The heatmap cell shows the mean over seeds.
            speedup[i, j] = float(np.mean(cell_speedups))
            seq_grid[i, j] = float(np.mean(cell_seq))
            par_grid[i, j] = float(np.mean(cell_par))
            print(
                f"{n:>7} {p:>8.4f} {avg_degree:>8.1f} {graph.num_edges:>9} "
                f"{seq_grid[i, j]:>9.3f} {par_grid[i, j]:>9.3f} {speedup[i, j]:>7.2f}x",
                flush=True,
            )

    return {
        "sizes": sizes,
        "probs": probs,
        "speedup": speedup,
        "seq": seq_grid,
        "par": par_grid,
        "points": points,
        "cpu_label": cpu_label,
    }


def plot_sweep(results: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sizes = results["sizes"]
    probs = results["probs"]
    speedup = results["speedup"]
    cpu_label = results["cpu_label"]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    vmax = max(2.0, float(np.nanmax(speedup)))
    vmin = min(float(np.nanmin(speedup)), 0.9)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # Cell-centered grid; use category indices so spacing is even regardless of
    # the (possibly geometric) sampling of n and p.
    nx, ny = len(sizes), len(probs)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("lightgrey")  # cells skipped by the blow-up guard
    mesh = ax.imshow(
        np.ma.masked_invalid(speedup),
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
    )

    # Annotate each cell with its speedup (or "skip" for guarded cells).
    for i in range(ny):
        for j in range(nx):
            val = speedup[i, j]
            label = "skip" if np.isnan(val) else f"{val:.2f}x"
            color = "grey" if np.isnan(val) else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(nx))
    ax.set_xticklabels(sizes)
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{p:g}" for p in probs])
    ax.set_xlabel("number of vertices  n")
    ax.set_ylabel("ER edge probability  p")
    ax.set_title(
        f"Parallel speedup of ECP flag (workers=-1 vs 1, {cpu_label})\n"
        "red = parallel wins, blue = parallel loses"
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("speedup (seq / par)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit P(y=1) = sigmoid(b0 + b1 x) by IRLS; return (beta, covariance).

    The covariance is the inverse Fisher information (X' W X)^-1, used for
    delta-method confidence intervals on the decision thresholds.
    """
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    hessian = np.eye(2)
    for _ in range(100):
        prob = _sigmoid(design @ beta)
        weights = np.clip(prob * (1.0 - prob), 1e-9, None)
        hessian = design.T @ (design * weights[:, None])
        gradient = design.T @ (y - prob)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(hessian)
    return beta, cov


def _prob_threshold_ci(
    beta: np.ndarray,
    cov: np.ndarray,
    prob: float,
    z: float,
) -> tuple[float, float, float]:
    """Sequential time at which P(parallel faster) == prob, with a CI.

    Solves b0 + b1 x = logit(prob) for x = log10(seq), propagates the parameter
    covariance through the delta method, and returns (time, lo, hi) in seconds.
    """
    logit = np.log(prob / (1.0 - prob))
    b0, b1 = beta
    x = (logit - b0) / b1
    grad = np.array([-1.0 / b1, -(logit - b0) / b1**2])
    se_x = float(np.sqrt(grad @ cov @ grad))
    return 10**x, 10 ** (x - z * se_x), 10 ** (x + z * se_x)


def plot_decision(
    results: dict,
    out_path: str,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> None:
    """Speedup vs. sequential runtime: the practical use-parallel decision rule.

    Every (n, p, seed) sample becomes one point at (sequential time, speedup).
    We model the binary outcome "parallel was faster" (speedup > 1) as a
    logistic function of log10(sequential time) and read off the 50% break-even
    time with a delta-method confidence interval from the fit covariance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = results["points"]
    cpu_label = results["cpu_label"]
    if not points:
        print("No points to plot for the decision figure.")
        return

    # Two-sided z multiplier for the requested CI level (e.g. 1.96 at 95%).
    ci_z = NormalDist().inv_cdf(1.0 - (1.0 - ci_level) / 2.0)

    seq = np.array([pt["seq"] for pt in points])
    su = np.array([pt["speedup"] for pt in points])
    wins = (su > 1.0).astype(float)  # 1 if parallel was faster

    log_seq = np.log10(seq)
    beta, cov = _fit_logistic(log_seq, wins)

    # The well-determined quantity: the 50% break-even seq time. Its confidence
    # interval bounds the break-even from above; the high-reliability tail is too
    # under-sampled to estimate usefully, so we don't draw it.
    levels = [
        (0.5, "dimgrey", "50% break-even"),
    ]
    thresholds = [(_prob_threshold_ci(beta, cov, q, ci_z), c, lbl) for q, c, lbl in levels]

    print(
        f"\nLogistic fit P(parallel faster) = sigmoid({beta[0]:.3f} "
        f"+ {beta[1]:.3f} * log10(seq))"
    )
    for (q, _, lbl), ((t, lo, hi), _, _) in zip(levels, thresholds, strict=True):
        factor = (hi / lo) ** 0.5  # symmetric multiplicative CI half-width
        print(
            f"  {lbl:<26} P={q:.2f}: {t:.2f}s  (x/ {factor:.2f}; "
            f"CI [{lo:.2f}, {hi:.2f}])"
        )

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(
        seq, su, color="tab:blue", s=45, alpha=0.7, edgecolor="black", linewidth=0.4,
        zorder=3, label="samples (left axis)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("sequential runtime, workers=1  (s, log scale)")
    ax.set_ylabel("speedup (seq / par)")

    # Logistic win-probability curve on a twin axis.
    ax2 = ax.twinx()
    grid = np.linspace(log_seq.min(), log_seq.max(), 200)
    prob = _sigmoid(beta[0] + beta[1] * grid)
    ax2.plot(10**grid, prob, color="tab:orange", lw=2.2, label="P(parallel faster)")
    ax2.set_ylabel("P(parallel faster)")
    ax2.set_ylim(-0.02, 1.02)

    # Threshold markers with confidence bands. The dotted probability guides run
    # only from the curve crossing out to the right axis (they belong to ax2).
    # Freeze the x-limits first so the guides reach the right spine exactly;
    # otherwise a draw-time autoscale widens the limits and leaves them short.
    ax.set_xlim(ax.get_xlim())
    xmax = ax.get_xlim()[1]
    for (q, color, lbl), ((t, lo, hi), _, _) in zip(levels, thresholds, strict=True):
        factor = (hi / lo) ** 0.5  # CI is symmetric in log-time: t x/ factor
        ax.axvspan(lo, hi, color=color, alpha=0.13)
        ax.axvline(
            t, color=color, lw=1.6, label=f"{lbl}: {t:.2f}s ($\\times/\\,{factor:.2f}$)"
        )
        ax2.hlines(q, t, xmax, color=color, ls=":", lw=0.9, alpha=0.7)

    ax.set_title(
        f"Use parallel when the sequential job is big enough ({cpu_label})\n"
        f"logistic fit of P(parallel faster); bands = {int(ci_level * 100)}% CI"
    )
    ax.grid(True, which="both", alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


def save_results(results: dict, path: str) -> None:
    """Persist measurements so figures can be regenerated without recomputing."""
    import json

    payload = {
        "sizes": results["sizes"],
        "probs": results["probs"],
        "cpu_label": results["cpu_label"],
        "speedup": results["speedup"].tolist(),
        "seq": results["seq"].tolist(),
        "par": results["par"].tolist(),
        "points": results["points"],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle)
    print(f"Saved data to {path}")


def load_results(path: str) -> dict:
    """Load measurements saved by :func:`save_results`."""
    import json

    with open(path) as handle:
        payload = json.load(handle)
    payload["speedup"] = np.array(payload["speedup"])
    payload["seq"] = np.array(payload["seq"])
    payload["par"] = np.array(payload["par"])
    payload.setdefault("cpu_label", f"{payload.get('cpu', '?')} threads")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000]
    )
    parser.add_argument(
        "--probs",
        type=float,
        nargs="+",
        default=[0.002, 0.004, 0.008, 0.016, 0.032],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="random seeds; each (n, p) cell is measured once per seed. More "
        "seeds give more decision-plot points and average out the heatmap.",
    )
    parser.add_argument(
        "--max-degree",
        type=float,
        default=1e9,
        help="skip (n, p) cells whose average degree exceeds this. The default "
        "computes every cell; lower it to guard against blow-up in the "
        "large-and-dense corner.",
    )
    parser.add_argument(
        "--heavy-degree",
        type=float,
        default=100.0,
        help="cells with average degree above this are measured with only the "
        "first seed (they are very slow); cheaper cells use all seeds.",
    )
    parser.add_argument(
        "--out", type=str, default=os.path.join(_DATA, "ecp_flag_sweep.png")
    )
    parser.add_argument(
        "--out-decision",
        type=str,
        default=os.path.join(_DATA, "ecp_flag_decision.png"),
        help="output path for the speedup-vs-sequential-time decision plot.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="recommend parallel once P(parallel faster) >= 1 - alpha.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="confidence level for the threshold confidence intervals.",
    )
    parser.add_argument(
        "--data", type=str, default=os.path.join(_DATA, "ecp_flag_sweep_data.json")
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="skip computation and regenerate figures from the saved --data file.",
    )
    args = parser.parse_args()

    if args.from_cache:
        results = load_results(args.data)
    else:
        results = run_sweep(
            args.sizes,
            args.probs,
            args.repeats,
            args.seeds,
            args.max_degree,
            args.heavy_degree,
        )
        save_results(results, args.data)
    plot_sweep(results, args.out)
    plot_decision(results, args.out_decision, alpha=args.alpha, ci_level=args.ci)


if __name__ == "__main__":
    main()
