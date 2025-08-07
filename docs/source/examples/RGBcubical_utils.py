import numpy as np


def pad_image(image, value):
    return np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=value)


def compute_RGB_contributions(image, inf_value=np.inf):
    # pad image
    image = pad_image(image, inf_value)

    # compute contributions of all cells,
    # starting from bottom left
    # uses lowert star filtration

    contributions = dict()

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            # lets track all the contributions
            # from cell i,j

            # itself, 2d cell
            f = tuple(image[i, j])
            contributions[f] = contributions.get(f, 0) + 1

            # 0d cell SW
            f = tuple(
                np.fmin(
                    image[i, j],
                    np.fmin(
                        image[i - 1, j - 1], np.fmin(image[i - 1, j], image[i, j - 1])
                    ),
                )
            )
            contributions[f] = contributions.get(f, 0) + 1

            # 1d cell W
            f = tuple(np.fmin(image[i, j], image[i, j - 1]))
            contributions[f] = contributions.get(f, 0) - 1

            # 1d cell S
            f = tuple(np.fmin(image[i, j], image[i - 1, j]))
            contributions[f] = contributions.get(f, 0) - 1

    # remove the contributions that are 0
    to_del = []
    for key in contributions:
        if contributions[key] == 0:
            to_del.append(key)
    for key in to_del:
        del contributions[key]

    return sorted(list(contributions.items()), key=lambda x: x[0])[:-1]


def EC_at_RGB_value(contributions, r, g, b):
    return sum(
        [
            c[1]
            for c in contributions
            if (c[0][0] <= r) and (c[0][1] <= g) and (c[0][2] <= b)
        ]
    )


def prune_contributions(contributions):

    total_ECP = dict()

    for a in contributions:
        total_ECP[a[0]] = total_ECP.get(a[0], 0) + a[1]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECP:
        if total_ECP[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECP[key]

    return sorted(list(total_ECP.items()), key=lambda x: x[0])


def difference_RGB_ECP(ecp_1, ecp_2, return_contributions=False):
    fmin = 0
    fmax = 257

    contributions = [((fmin, fmin, fmin), 0), ((fmax, fmax, fmax), 0)]

    contributions += ecp_1
    contributions += [(c[0], -1 * c[1]) for c in ecp_2]

    contributions = (
        [((fmin, fmin, fmin), 0)]
        + prune_contributions(contributions)
        + [((fmax, fmax, fmax), 0)]
    )

    #     print(contributions)

    R_list = sorted(set([c[0][0] for c in contributions]))
    G_list = sorted(set([c[0][1] for c in contributions]))
    B_list = sorted(set([c[0][2] for c in contributions]))

    difference = 0

    for i, r in enumerate(R_list[:-1]):
        delta_r = R_list[i + 1] - R_list[i]
        for j, g in enumerate(G_list[:-1]):
            delta_g = G_list[j + 1] - G_list[j]
            for z, b in enumerate(B_list[:-1]):
                delta_b = B_list[z + 1] - B_list[z]

                difference += abs(
                    EC_at_RGB_value(contributions, r, g, b)
                    * delta_r
                    * delta_g
                    * delta_b
                )

    if return_contributions:
        return difference, contributions
    else:
        return difference


def approximate_difference_RGB_ECP(ecp_1, ecp_2):
    m_1 = np.zeros((257, 257, 257))
    m_2 = np.zeros((257, 257, 257))

    for c in ecp_1:
        m_1[c[0][0] :, c[0][1] :, c[0][2] :] += c[1]

    for c in ecp_2:
        m_2[c[0][0] :, c[0][1] :, c[0][2] :] += c[1]

    difference = np.sum(np.abs(m_1 - m_2))

    return difference
