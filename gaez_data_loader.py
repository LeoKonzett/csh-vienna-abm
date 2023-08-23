# Imports

import numpy as np
from libtiff import TIFF


class GlobalAezV4:
    """Loader class for the Global Agro-Ecological Zones (GAEZ) version 4 data set published by UN FAO."""
    _array = np.array([0])
    _distance_array = np.array([0.])
    _radius_eq, _radius_to_centre = 40_000, 6371  # approx. in km # TODO: These are constants, define as such.
    _csize_km, _csize_deg = 0, 0
    _nrows, _ncols = 0, 0  # for window of interest - zero if no woi
    _lat_start, _long_start = -90, -180  # in degrees

    def load(self, fp, verbose=True, idx=0):
        """Load TIF data as image using LIBTIFF.
            Contains multiple images -> pick highest resolution (i.e. idx = 0) by default. """
        img_container = TIFF.open(fp).iter_images()  # returns generator object

        count = 0
        while count <= idx:
            self._array = next(img_container)
            count += 1

        # get latitude cell size
        self._csize_km = self._radius_eq / self._array.shape[1]
        self._csize_deg = 360 / self._array.shape[1]

        # set cells and rows
        self._nrows, self._ncols = self._array.shape

        if verbose:
            # num_images = sum(1 for _ in img_container) - this doesn't work as generators cannot be rewound
            # print(f"TIF container has {num_images} images")
            print("\n Loaded image has resolution: ", self._array.shape)
            print("\n Latitude cell size [km] for that image is: ", self._csize_km)
            print("\n Latitude cell size [deg] for that image is: ", self._csize_deg)

    def resize(self, target_size=(10, 10)):
        """ Resize matrix to target size by averaging cell values."""
        target_n_rows, target_n_cols = target_size
        n_rows, n_cols = self._array.shape
        scale_row, scale_col = n_rows // target_n_rows, n_cols // target_n_cols

        # Reshape and average
        self._array = self._array.reshape(target_n_rows, scale_row, target_n_cols, scale_col).mean(axis=(1, 3))

    def set_woi(self, r0=0, c0=0, n_rows=100, n_cols=100):
        """ Select window of interest in row / column format.
        Array indexing starts from upper left corner, which is latitude = + 90 and longitude = -180 (degrees)"""
        self._nrows, self._ncols = n_rows, n_cols

        # set reference point (upper left - northwest) for latitude coordinates
        self._lat_start = 90 - r0 * self._csize_deg
        self._long_start = -180 + c0 * self._csize_deg

        self._array = self._array[r0:r0 + n_rows, c0:c0 + n_cols]

    def get_distance_matrix(self, verbose=False):
        """
        goal: we have a (n_cols, n_rows, n_rows) matrix. If we query with e.g. index (0, 0, 0),
        we get the distance for relative long. diff 0, and for latitude vals self._xul, self._xul
        instead of np.meshgrid(), we use numpy broadcasting to save memory
        """
        latitudes = np.linspace(self._lat_start, self._lat_start + (self._nrows - 1) * self._csize_deg, self._nrows)
        longitudes_diff = np.linspace(0, (self._ncols - 1) * self._csize_deg,
                                      self._ncols)  # only relative values needed

        # prepare for broadcasting
        longitudes_diff = longitudes_diff[:, None, None]
        lat_vals_x = latitudes[None, :, None]
        lat_vals_y = latitudes[None, None, :]

        # get distance - problem: We have a memory error but broadcasting works like a charm
        num_elements = self._ncols * self._nrows ** 2
        if num_elements > pow(10, 8):  # raise error if matrix takes more than ~ 100 MB
            raise MemoryError("Not enough memory can be allocated, consider a smaller window of interest.")

        # convert to radians
        lat_vals_x *= np.pi / 180
        lat_vals_y *= np.pi / 180
        longitudes_diff *= np.pi / 180

        # apply haversine formula
        a = np.sin((lat_vals_x - lat_vals_y) / 2) ** 2 + \
            np.cos(lat_vals_x) * np.cos(lat_vals_y) * np.sin(longitudes_diff / 2) ** 2
        self._distance_array = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        self._distance_array *= self._radius_to_centre

        if verbose:
            print("\n Distance matrix has shape ", self._distance_array.shape,
                  "\n Window of Interest has columns and rows ", self._ncols, self._nrows)

    def get_arc_distance(self, p_origin, p_targets_r, p_targets_c):
        """ To come. No lat / long conversion needed as the distance matrix takes care of that."""
        r0, c0 = p_origin

        # longitude is column, latitude is row
        c_deltas = np.abs(c0 - p_targets_c)  # only relative difference matters

        # first axis is relative longitudinal difference, second axis is latitude 1, third axis is latitude 2
        distances = self._distance_array[c_deltas, r0, p_targets_r]  # c0 gets broadcast to match the array shapes

        return distances

    def get_env(self, water_var=0):
        """ get environment based on gaez v4 data set (33 AEZ classes, 5 arc-minute resolution)
        input data is an integer array with entries {0, 33}, where 0 is water (water_var) and e.g. 32 is built-up land.
        precise docs can be found at Gaez V4 user guide, page 162.
        output data is a 3D array with exactly one non-zero entry along the last axis that denotes
        the AEZ class to which the village belongs.
        """
        input_arr = self._array
        variables = np.unique(input_arr)  # different AEZ classes
        num_env_vars = variables.size

        env = np.zeros([*input_arr.shape, num_env_vars], dtype=float)
        for idx, val in enumerate(variables):  # start from 1 to skip water entries
            mask = input_arr == val
            env[mask, idx] = 1

        # if successful, sum along last axis is unity
        assert np.all(np.sum(env, axis=-1) == 1), f"sum is non-unity and is {np.sum(env, axis=-1)}"

        # handle water - can be extended to other variables
        if water_var in variables:
            print("In Lattice.init_env_from_gaez(): Input environment contains water. Set to zero.")
            is_water = input_arr == water_var
            water_idx = np.squeeze(np.argwhere(variables == water_var))
            env[is_water, water_idx] = 0

        return env
