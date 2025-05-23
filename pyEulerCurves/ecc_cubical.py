import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ._compute_local_EC_cubical import (
    compute_contributions_two_slices,
    compute_contributions_two_slices_PERIODIC,
)


import time


def compute_contributions_single_slice(slices, dim, periodic_boundary):
    if type(periodic_boundary) is list:
        if len(periodic_boundary) != len(dim):
            raise ValueError(
                "Dimension of input is different from the number of boundary conditions"
            )
        return dict(
            compute_contributions_two_slices_PERIODIC(slices, dim, periodic_boundary)
        )
    else:
        return compute_contributions_two_slices(slices, dim)


def compute_cubical_contributions(
    top_dimensional_cells, dimensions, periodic_boundary=False, workers=1, chunksize=100
):
    slice_len = 1
    for d in dimensions[:-1]:
        slice_len *= d

    # number of filtration parameters
    num_f = len(top_dimensional_cells[0])

    # add padding
    top_dimensional_cells = np.concatenate(
        (
            [[np.inf] * num_f for i in range(slice_len)],
            top_dimensional_cells,
            [[np.inf] * num_f for i in range(slice_len)],
        )
    )

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        ECC_list = executor.map(
            compute_contributions_single_slice,
            [
                top_dimensional_cells[i : i + 2 * slice_len]
                for i in range(0, len(top_dimensional_cells) - slice_len, slice_len)
            ],
            itertools.repeat(dimensions[:-1] + [2]),
            itertools.repeat(periodic_boundary),
            chunksize=chunksize,
        )

    end = time.perf_counter()
    print("Parallel part done")
    print(f"Elapsed time: {end - start:.1f} seconds")

    start = time.perf_counter()
    ECC_dict = dict()
    counter = 0
    for single_ECC in ECC_list:
        counter += 1
        for key, item in single_ECC:
            k = tuple(key)
            ECC_dict[k] = ECC_dict.get(k, 0) + item

    # remove the contributions that are 0
    to_del = []
    for key in ECC_dict:
        if ECC_dict[key] == 0:
            to_del.append(key)

    for key in to_del:
        del ECC_dict[key]

    end = time.perf_counter()
    print("Merged {} dicts".format(counter))
    print(f"Elapsed time: {end - start:.1f} seconds")

    return list(ECC_dict.items())
