import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ._compute_local_EC_cubical import compute_contributions_two_slices, compute_contributions_two_slices_PERIODIC



def compute_contributions_single_slice(slices, dim, periodic_boundary):
    if type(periodic_boundary) is list:
        if len(periodic_boundary) != len(dim):
            raise ValueError(
                "Dimension of input is different from the number of boundary conditions"
            )
        return dict(compute_contributions_two_slices_PERIODIC(slices, dim, periodic_boundary))
    else:
        return dict(compute_contributions_two_slices(slices,dim))


def compute_cubical_contributions(top_dimensional_cells, dimensions,
                                  periodic_boundary=False, workers=1):

    slice_len = 1
    for d in dimensions[:-1]:
        slice_len *= d

    #add padding
    top_dimensional_cells = np.concatenate( ([np.inf for i in range(slice_len)],
                                             top_dimensional_cells,
                                             [np.inf for i in range(slice_len)]) )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        ECC_list = executor.map(compute_contributions_single_slice,
                                 [top_dimensional_cells[i:i+2*slice_len]
                                         for i in range(0,
                                                        len(top_dimensional_cells)-slice_len,
                                                        slice_len)],
                                 itertools.repeat(dimensions[:-1]+[2]),
                                 itertools.repeat(periodic_boundary)
                                 )

    ECC_dict = dict()
    for single_ECC in ECC_list:
        for key in single_ECC:
            ECC_dict[key] = ECC_dict.get(key, 0) + single_ECC[key]

    # remove the contributions that are 0
    to_del = []
    for key in ECC_dict:
        if ECC_dict[key] == 0:
            to_del.append(key)

    for key in to_del:
        del ECC_dict[key]

    ecc = sorted(list(ECC_dict.items()), key = lambda x: x[0])

    return np.array(ecc[:-1])
