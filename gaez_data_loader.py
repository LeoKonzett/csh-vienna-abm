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

        # longitude is row, latitude is column
        r_deltas = np.abs(r0 - p_targets_r)  # only relative difference matters

        # first axis is relative longitudinal difference, second axis is latitude 1, third axis is latitude 2
        distances = self._distance_array[r_deltas, c0, p_targets_c]  # c0 gets broadcast to match the array shapes

        return distances