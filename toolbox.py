import numpy as np


def remove_all_indices(arr, target_index):
    """Remove indices that match target_index from array. OUTDATED"""
    mask = np.any(arr != target_index, axis=1)
    return arr[mask]


def remove_entry_from_list(from_list, target):
    """Removes ALL entries that are equal to target from a custom (nested) list"""
    if not isinstance(target, tuple):
        raise TypeError("Tuple needed")
    return [idx for idx in from_list if idx != target]


def flip_entries(array, p_flip=0.01):
    """ Flip values of a binary array with probability p_flip."""
    random_numbers = np.random.random(size=array.shape)  # Generate floats in interval [0, 1)
    flip = random_numbers < p_flip
    return np.logical_xor(array, flip).astype(int)  # Convert boolean to integer


def calculate_productivity(skill_arr, env_arr, prod_scaling=1, min_prod=0):
    """ Calculates the productivity according to prod_scale * <S_i, E_i>.
    If skill_arr and env_arr are array_like with dimension > 1, calculate dot product along the last axis.
    Else, return dot product directly.
    The return value is multiplied with prod_scaling and clipped to clip_min if not None
    """
    if len(env_arr.shape) == 1:  # 1D array
        arr = np.dot(env_arr, skill_arr)

    elif len(env_arr.shape) == 2:  # env is 2D and skill is 1D - use np.dot along last axis
        assert len(skill_arr.shape) == 1
        arr = np.dot(env_arr, skill_arr)  # ordering is important!! env is N-D, skill is 1-D

    elif len(env_arr.shape) == 3:  # 3D array
        arr = np.einsum("ijk, ijk -> ij", skill_arr, env_arr)

    else:
        raise TypeError("Only implemented for 1-3D arrays")

    arr *= (prod_scaling / env_arr.shape[-1])  # scale and normalize
    if min_prod > 0:
        np.clip(arr, min_prod, None, out=arr)  # clip to minimal value

    assert not np.any(np.isnan(arr)), "Productivity array contains NaNs"
    assert not np.any(np.isinf(arr)), "Productivity array contains +/- infinity"

    return arr


def get_distribution(inputs, mn=None):
    """Calculate a probability distribution based on rescaling a nd-array inputs between 0 and 1."""
    # min - max scaling
    mx = np.amax(inputs)
    if min is None:
        mn = np.amin(inputs)

    if mx == mn:  # min-max scaling yields NaN - assign uniform distribution
        probabilities = None
    else:  # rescale to [0, 1] range
        probabilities = (inputs - mn) / (mx - mn)
        probabilities = probabilities / np.sum(probabilities)

    return probabilities


def set_neighborhood(distance=1, neigh_type="moore"):
    """
    General: Return possible neighbor candidates for a cell at location "cell_location".
    Variables to consider are radius (for a network), or type of neighborhood + distance of neighborhood (for lattice)
    For now we only consider a lattice.

    The current idea is to avoid a for loop here, i.e. we want to find a list of indices that we can slice the array.
    The problem is that the indices possibly overlap, and we can't slice the array twice. Other, we have a solution
    using np.newaxis and numpy broadcasting magic.
    So it probably *is* better to use a for loop for now.

    Specific: Return the indices for a von-Neumann (taxicab metric) or Moore (incl. diagonals) neighborhood.
    The output of this function (x and y indices) are to be added to the desired point. Only used once and then stored.
    This returns a transposed index array which might not be ideal. Fix.
    """
    idx_range = np.arange(-distance, distance + 1)  # assumes centered point
    r_indices, c_indices = np.meshgrid(idx_range, idx_range, indexing="ij")  # use ij convention

    r_indices = r_indices.ravel()  # flatten
    c_indices = c_indices.ravel()

    # an array of length N has its center at (N-1) // 2 -> 2D center at (N-1) / 2 x (N + 1)
    ele_center = distance * (2 * distance + 2)
    r_indices = np.delete(r_indices, ele_center)  # delete center element
    c_indices = np.delete(c_indices, ele_center)

    if neigh_type == "moore":
        return r_indices, c_indices

    elif neigh_type == "von_neumann":
        to_keep = []
        for idx, (rr, cc) in enumerate(zip(r_indices, c_indices)):
            if (np.abs(rr) + np.abs(cc)) <= distance:
                to_keep.append(idx)

        return r_indices[to_keep], c_indices[to_keep]

    else:
        raise NotImplementedError("Valid methods names are moore and von_neumann.")
