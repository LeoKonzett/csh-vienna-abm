import numpy as np
import noise
from tqdm import tqdm
import toolbox


class Lattice:
    """Base object for population dynamics in medieval Europe. Square lattice.
    Central object is a square lattice with shape (time, x, x) which keeps the track of the population in each village.
    """
    _gaez_instance = None  # instance of GlobalAezV4 in gaez_data_loader.py

    def __init__(self, num_iter=5_000, seed=50, idx_start=(0, 0), population_growth_rate=1 / 30,
                 max_productivity=200, n_rows=10, n_cols=10, starting_pop=10):
        """ Initializes lattice object with shape (num_iter, n_rows, n_cols). assumes that env. vector
        and skill vector have the same length. Also initializes environment and skill vectors to calculate productivity.
        Current data type of population array is float (integer messes up the update rule)
        """
        self.rng = np.random.default_rng(seed=seed)
        self.r0, self.c0 = idx_start  # location of first settlement
        self.shape = (num_iter, n_rows, n_cols)
        self.n_rows, self.n_cols = n_rows, n_cols
        self.population = np.zeros(self.shape)
        self.starting_pop = starting_pop
        self.population[0, self.r0, self.c0] = starting_pop

        self.population_growth_rate = population_growth_rate
        self.max_productivity = max_productivity  # maximum carrying capacity ( = productivity)
        self.prod = np.zeros((n_rows, n_cols))
        self.prod_min = 0  # minimum carrying capacity that a cell has regardless of environment
        self.clip_population_to_zero = True  # clip negative population to zero

        # Environment parameters
        self.env = np.array([0])  # environment placeholder variable
        self.num_env_vars = 0  # number of distinct environments (e.g. very hot, very cold, very wet, arid)
        self.num_skill_vars = 0  # corresponding number of skills
        self.skills = np.array([0])
        self.productivity_step_size = None  # productivity varies from 0 to n * self.productivity_step_size
        self.migration_thresh = None  # min. productivity for villagers to migrate

        self.num_iter = 1
        self.is_empty = self.population[0, :, :] == 0  # flag empty cells
        self.is_uninhabitable = np.array([False])  # set in method self.load_env

        # Simulation speed parameters
        self.use_valid_neighs_flag = False
        self.has_valid_neighs = np.array([True])

        # Fission parameters
        self.pop_min, self.pop_max = None, None  # if population above pop_max -> fission with p = 1
        self.indices_r, self.indices_c = [], []  # indices to select neighbors
        self.search_intelligently = True  # If True, select cells probabilistically based on productivity
        self.max_distance = None  # maximum distance between two points in km
        self.fission_distribution_type = "linear"

        # Mutation parameters
        self.env_mutation_rate = None  # Probability that one entry of the environment vector per site flips
        self.skill_mutation_rate = None  # Same but for skills
        self.mutation_method = None

        # Params for Metropolis mutation
        self.metropolis_scale = None  # scaling factor for metropolis algorithm

        # Params for adaptive mutation
        self.min_pop_mutation, self.max_pop_mutation = None, None
        self.p_gain_u, self.p_gain_nu = None, None
        self.p_lose_u, self.p_lose_nu = None, None
        self.mutation_distribution_type = "constant"

        self.geo_constraints = None  # geography data that is used to modulate the carrying capacity

    def load_env(self, env=None, kind="random", num_env_vars=None, correlation_length=None, verbose=False):
        """Load either a random environment (perlin noise) or a custom environment. The custom
        environment needs to be in the correct format (i.e. matching shapes, and having entries
        that are either 0 or 1).
        Perlin noise yields a random environment that is spatially correlated. The correlation length
        is controlled by a scale parameter.
        The custom environment needs to be 3-dimensional (rows, cols, num_env_vars).
        """
        if kind == "random":  # load Perlin environment
            assert num_env_vars is not None, "Number of mutually exclusive environment areas required."
            assert correlation_length is not None, "Spatial correlation length required for Perlin Noise."
            self.num_env_vars = num_env_vars

            self.env = np.zeros((self.n_rows, self.n_cols, num_env_vars))
            for ii in range(self.n_rows):
                for jj in range(self.n_cols):
                    self.env[ii, jj] = [noise.pnoise3(ii * correlation_length, jj * correlation_length,
                                                      k * correlation_length) for k in range(num_env_vars)]

            for ii in range(num_env_vars):  # convert to 0 or 1
                self.env[:, :, ii] = np.where(self.env[:, :, ii] > 0, 1, 0)

        elif kind == "custom":  # load custom environment
            assert env is not None, "Provide environment to load"
            assert env.shape[:2] == (self.n_rows, self.n_cols), "Number of rows / columns doesn't match"
            self.num_env_vars = env.shape[2]
            if verbose:
                print(f"In meth: lattice.load_env: The environment has {self.num_env_vars} mutually exclusive types.")

            assert np.all((env == 0) | (env == 1)), "Environment needs to consist of either zeros or ones."

            self.env = env
            if np.all(self.env[self.r0, self.c0, :] == 0):
                raise ValueError("Environment at start site consists of all zeros (i.e. water). Choose different site.")

        else:
            raise NotImplementedError("Possible environment loaders are -custom- and -random-")

        # Initialize corresponding skill array after environment is fixed
        self.num_skill_vars = self.num_env_vars
        self.skills = np.zeros_like(self.env)
        self.skills[self.r0, self.c0] = self.rng.integers(size=self.num_skill_vars, low=0, high=1, endpoint=True)

        # Calculate productivity step size
        self.productivity_step_size = self.max_productivity / self.num_env_vars

        # Set uninhabitable cells - sites where environment vector is all zero can never be populated
        self.is_uninhabitable = np.all(self.env == 0, axis=-1)

    def set_fission_rules(self, bounds, migration_threshold=None, include_diagonals=False, search_distance_pixels=1,
                          search_intelligently=False, max_distance_km=None, dist_type="linear",
                          params_are_relative=False):
        """
        Sets probability distribution that governs village fission.
        :param bounds: tuple (min, max) - enforces p_split(N < min) = 0 and p_split(N > max) = 1
        :param migration_threshold: int - sites below this threshold aren't settled
        :param include_diagonals: bool - if True, include diagonals when searching for new sites
        :param search_distance_pixels: int - allowed distance in horizontal / vertical pixels between origin and new site
        :param search_intelligently: bool - if True, calculate probability distributions for search
        :param max_distance_km: int or None - if not None, use km instead of pixel distances
        :param dist_type: str - distribution type between the thresholds set by bounds
        :param params_are_relative: bool - if True, multiply input vals by self.productivity_step_size
        """
        neigh_type = "moore" if include_diagonals else "von_neumann"
        self.indices_r, self.indices_c = toolbox.set_neighborhood(distance=search_distance_pixels,
                                                                  neigh_type=neigh_type)

        if migration_threshold is not None:
            assert migration_threshold > 0, "Current version requires a migration threshold larger than 0"
            self.migration_thresh = migration_threshold
        else:  # migrate if at least one skill matches
            self.migration_thresh = np.floor(self.productivity_step_size)

        self.search_intelligently = search_intelligently  # uniform distribution or not

        if max_distance_km is not None:  # TODO: This should be a sub-class
            assert self._gaez_instance is not None, "Requires instance of class gaez_data_loader"
            self.max_distance = max_distance_km
            print(f"Villagers have maximum search radius of {max_distance_km} kilometres.")

        assert bounds[1] > bounds[0], "Minimum needs to be smaller than maximum."
        self.pop_min, self.pop_max = bounds

        if (self.productivity_step_size * self.num_env_vars) < self.pop_max:
            raise ValueError("Currently, fully grown cells do not split with p=1.")

        assert dist_type in ["linear", "sigmoid"], "Valid distribution types are sigmoid and linear."
        self.fission_distribution_type = dist_type

        if params_are_relative:  # choose parameters based on productivity step size
            self.pop_min *= np.floor(self.productivity_step_size)
            self.pop_max *= np.floor(self.productivity_step_size)
            self.migration_thresh *= np.floor(self.productivity_step_size)

    def load_geo_constraints(self, arr_constraints):
        """ Load an array to modulate the carrying capacity (productivity).
        The array must contain values between 0 and 1, where 1 denotes a maximum modulation
        and 0 denotes no modulation. """
        assert np.all(arr_constraints.shape == (self.shape[1], self.shape[2])), "Shapes don't match"
        assert np.all((0 <= arr_constraints) & (arr_constraints <= 1)), "Provide an array with values between 0 and 1"
        self.geo_constraints = arr_constraints

    def load_gaez_instance(self, gaez_instance):
        """load an instance of Global_AEZ data. Is pass-by-reference"""
        self._gaez_instance = gaez_instance
        # TODO: Don't access private variables
        assert gaez_instance._nrows == self.shape[1] and gaez_instance._ncols == self.shape[2]

    def set_mutation_rules(self, env_mutation_rate=None, skill_mutation_rate=None,
                           skill_mutation_method="adaptive", metropolis_scale=None,
                           params_adaptive=None):
        """Sets / updates parameters for mutation such as environment / skill mutation rate. Other parameters are
        - scaling ratio for metropolis mutation strategy
        - acceptance probabilities for the adaptive mutation strategy
        - rate modulation for the adaptive mutation strategy - distribution_type
        """
        assert skill_mutation_method in ["random", "metropolis", "adaptive"], \
            "Valid methods are random, metropolis, and adaptive"
        self.mutation_method = skill_mutation_method
        # Note: In the current implementation, only the adaptive mutation method can use a non-uniform distribution
        # TODO: Fix

        if env_mutation_rate is not None:
            assert 0 <= env_mutation_rate <= 1, "Rate is probability and should be between 0 and 1"
        if skill_mutation_rate is not None:
            assert 0 <= skill_mutation_rate <= 1
        self.env_mutation_rate = env_mutation_rate
        self.skill_mutation_rate = skill_mutation_rate

        if skill_mutation_method == "adaptive":
            # Check keys
            assert params_adaptive is not None, "Provide a dictionary with parameters"
            keys_required = {"p_gain_useful", "p_gain_useless", "p_lose_useful",
                             "p_lose_useless", "dist_type", "n_min", "n_max"}
            keys_optional = keys_required | {"bounds_are_relative"}
            if not keys_required <= params_adaptive.keys() <= keys_optional:
                raise ValueError(f"Include all required keys: {keys_required} and do not include "
                                 f"keys that are not optional: {keys_optional}")

            # Set variables
            assert params_adaptive.get("dist_type") in ["constant", "linear", "sigmoid"]
            self.mutation_distribution_type = params_adaptive["dist_type"]
            self.p_gain_u, self.p_gain_nu = params_adaptive["p_gain_useful"], params_adaptive["p_gain_useless"]
            self.p_lose_u, self.p_lose_nu = params_adaptive["p_lose_useful"], params_adaptive["p_lose_useless"]

            # Check bounds
            if params_adaptive["dist_type"] != "constant":
                self.min_pop_mutation, self.max_pop_mutation = params_adaptive["n_min"], params_adaptive["n_max"]
                assert self.min_pop_mutation < self.max_pop_mutation, "Minimum needs to larger than maximum."
                if self.min_pop_mutation < self.starting_pop:
                    raise RuntimeWarning("At t = 0, the initial village has zero probability to mutate a skill")

                if params_adaptive["bounds_are_relative"]:
                    self.min_pop_mutation *= np.floor(self.productivity_step_size)
                    self.max_pop_mutation *= np.floor(self.productivity_step_size)

        if skill_mutation_method == "metropolis":
            assert metropolis_scale is not None
            self.metropolis_scale = metropolis_scale  # scaling ratio for Metropolis

    def set_speed_params(self):
        """ Speed-up possible if: 1 - environment must be static 2 - skill mutation must be adaptive
        # 3 - probability to lose a useful skill must be 0 """
        if self.env_mutation_rate is None and self.mutation_method == "adaptive":
            if self.p_lose_u == 0:
                self.use_valid_neighs_flag = True
                self.has_valid_neighs = np.ones([self.n_rows, self.n_cols], dtype=bool)

        else:
            print("Speed-up is only possible if environment ist static, if skill mutation method is adaptive, "
                  "and if the probability to accept a flip that would lead to losing a useful skill is zero")

    def mutate_skill_metropolis(self, p_flip, mask_additional=None, scale=1):
        """Metropolis-like mutation for skills. Compares the ratio alpha = Prod(flip) / prod_previous and accepts
        the proposed flip if alpha = num where num is a uniform float in [0, 1]
        If scale != 1, the ratio array gets exponentiated by scale, e.g. ratio -> ratio ** scale
        This exacerbates larger differences, e.g. 0.75 -> 0.56 (scale = 2) -> 0.42
        """
        prod_previous = toolbox.calculate_productivity(self.skills, self.env, prod_scaling=self.max_productivity,
                                                       min_prod=self.prod_min)

        skills_update = self.flip_single_entry_per_cell(self.skills, p_flip=p_flip, mask_additional=mask_additional)
        prod_update = toolbox.calculate_productivity(skills_update, self.env, prod_scaling=self.max_productivity,
                                                     min_prod=self.prod_min)
        with np.errstate(divide="ignore", invalid="ignore"):  # ignore division by zero warning
            ratio = np.divide(prod_update, prod_previous)

        ratio = np.power(ratio, scale)  # apply custom scaling

        nums = self.rng.uniform(low=0, high=1., size=self.prod.shape)
        accept_flip = ratio > nums  # accept flip dep. on ratio

        self.skills[accept_flip] = skills_update[accept_flip]

        return self.skills

    def mutate_skill_diff_ratios(self):
        """Skill mutation for adaptive scenario. We identify 4 different cases:
        Gain of useful skill. Loss of useful skill. Gain of useless skill.
        Loss of useless skill. We assign different probabilities for all 4 cases.
        The variable modulation determines how the mutation base rate is modulated wrt. population density.
        Linear means that the mutation rate is proportional to the number of villagers.
        Sigmoid means that the mutation rate is a sigmoid function of # villagers.
        """
        nums = self.rng.uniform(low=0, high=1, size=self.prod.shape)

        # this function is called *before* updating the population per step and hence uses the population from
        # the previous iteration
        population = self.population[self.num_iter - 1]
        mutation_probs = toolbox.get_modulated_distribution(population,
                                                            bounds=(self.min_pop_mutation, self.max_pop_mutation),
                                                            base_rate=self.skill_mutation_rate,
                                                            kind=self.mutation_distribution_type)

        # get r/c indices of cells for which a flip happens - cells must be occupied
        mask = (nums < mutation_probs) & np.logical_not(self.is_empty)
        idx_r, idx_c = np.nonzero(mask)

        # pick a random number for each flipped cell
        idx_flip = self.rng.choice(range(self.num_env_vars), size=idx_r.size)

        # extract relevant values
        env_vals = self.env[idx_r, idx_c, idx_flip]
        skill_vals = self.skills[idx_r, idx_c, idx_flip]

        # get acceptance probabilities
        m1 = (skill_vals == 0) & (env_vals == 0)  # gain skill that isn't used
        m2 = (skill_vals == 0) & (env_vals == 1)  # gain skill that is used
        m3 = (skill_vals == 1) & (env_vals == 0)  # lose skill that isn't used

        probabilities = np.select([m1, m2, m3], [self.p_gain_nu, self.p_gain_u, self.p_lose_nu],
                                  default=self.p_lose_u)
        nums = self.rng.uniform(low=0, high=1, size=env_vals.size)
        mask_1d = nums < probabilities
        
        # enable accepted flips and replace
        replacement_vals = np.logical_xor(skill_vals, mask_1d).astype(int)
        self.skills[idx_r, idx_c, idx_flip] = replacement_vals

    def flip_single_entry_per_cell(self, array, p_flip=0.01, mask_additional=None):
        """Flip one randomly selected entry (uniform pdf with rng) with probability p_flip.
        Env has shape (N, N, 10) - we need NxN random floats in [0, 1].
        E.g. for a NxN lattice, we have one expected flip per function call if p_flip = 1 / (NxN)
        If mask is not None, flip only where Mask is True (in addition to random draw)
        """
        nums = self.rng.uniform(low=0, high=1, size=self.prod.shape)
        mask = nums < p_flip  # select cells for which a flip happens

        if mask_additional is not None:
            assert mask.dtype == bool, "Requires boolean mask"
            mask = mask & mask_additional

        # select number between 0 and num_env_vars for each cell
        flip_indices = self.rng.choice(range(self.num_env_vars), size=self.prod.shape)

        # flip values where mask is True - alternative is np.ix_() for easy broadcasting
        rows, columns = np.arange(self.shape[1]), np.arange(self.shape[2])
        replacement_vals = np.logical_xor(array[rows[:, np.newaxis], columns, flip_indices], mask).astype(int)

        # copy values - else caller also sees modification
        array_flip = np.copy(array)
        array_flip[rows[:, np.newaxis], columns, flip_indices] = replacement_vals

        if np.any(mask):  # assert that array hasn't been modified in-place if at least one entry is flipped
            assert np.any(array != array_flip), "In meth: flip_single_entry_per_cell: Array has been modified in-place."

        return array_flip

    def move_forward(self):
        """ Updated move_forward method.
         NB: If the number of villages that want to split exceeds a threshold, we apply a reverse search
         strategy. That is, we look for **empty** villages that have occupied neighbors, and select a cell that
         will split. """
        # run mutations
        if self.env_mutation_rate is not None:
            self.env = self.flip_single_entry_per_cell(self.env, p_flip=self.env_mutation_rate)
            self.is_uninhabitable = np.all(self.env == 0, axis=-1)  # update un-inhabitable cells upon env mutation

        if self.skill_mutation_rate is not None:
            if self.mutation_method == "random":
                is_occupied = ~self.is_empty
                self.skills = self.flip_single_entry_per_cell(self.skills, p_flip=self.skill_mutation_rate,
                                                              mask_additional=is_occupied)
            elif self.mutation_method == "metropolis":
                is_occupied = ~self.is_empty
                self.skills = self.mutate_skill_metropolis(p_flip=self.skill_mutation_rate,
                                                           mask_additional=is_occupied, scale=self.metropolis_scale)
                
            elif self.mutation_method == "adaptive":
                self.mutate_skill_diff_ratios()
            else:
                raise Exception("Mutation method not implemented. Valid methods are random, metropolis, and 4_rates.")

        # calculate productivity to update village population
        self.prod = toolbox.calculate_productivity(self.skills, self.env,
                                                   prod_scaling=self.max_productivity, min_prod=self.prod_min)

        # modulate productivity according to geographic constraints
        if self.geo_constraints is not None:
            self.prod = self.prod * (1 - self.geo_constraints)

        # update population - empty cells will stay empty regardless of productivity
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.population[self.num_iter - 1, self.r0, self.c0] < 1:
                self.population[self.num_iter - 1, self.r0, self.c0] = 1  # ensure one villager remains

            self.population[self.num_iter] = self.population[self.num_iter - 1] + \
                self.population_growth_rate * self.population[self.num_iter - 1] \
                * (1 - np.divide(self.population[self.num_iter - 1], self.prod))

        if self.clip_population_to_zero:  # set NaN or negative populations to zero
            mask = np.isnan(self.population[self.num_iter]) | (self.population[self.num_iter] < 0)
            self.population[self.num_iter][mask] = 0

        if np.sum(self.population[self.num_iter]) == 0:  # handle empty lattice
            raise RuntimeError("All villagers died (empty lattice). This occurs if the carrying capacity at the "
                               "starting site is zero, or turns zero after mutations.")

        # Get split probability - based on global float draw
        prob_to_split = toolbox.get_modulated_distribution(self.population[self.num_iter],
                                                           bounds=(self.pop_min, self.pop_max),
                                                           kind=self.fission_distribution_type)
        floats = self.rng.uniform(low=0.0, high=1.0, size=self.prod.shape)

        # Get empty cells after population update (incl. NaN and negative values that have been set to zero)
        self.is_empty = self.population[self.num_iter] == 0

        # Count the number of cells that want to split and set helper variable
        num_cells_splitting = np.sum(prob_to_split > floats)

        # If more than half of all habitable cells want to split, use reverse search strategy
        thresh = (self.n_rows * self.n_cols - np.sum(self.is_uninhabitable)) / 2
        if num_cells_splitting <= thresh:
            search_empty_cells = True
        else:
            search_empty_cells = False

        cells_that_split = prob_to_split > floats
        if search_empty_cells:  # villagers that move to empty locations
            # Here: Add mask that says if cell has valid neighbors
            if self.use_valid_neighs_flag:
                idx_r, idx_c = np.nonzero(cells_that_split & self.has_valid_neighs)
            else:
                idx_r, idx_c = np.nonzero(cells_that_split)
        else:  # reverse search
            idx_r, idx_c = np.nonzero(self.is_empty & (~self.is_uninhabitable))  # look for empty and habitable cells

        if idx_r.size > 0:
            # !! the method migrate_to modifies self.is_empty. This is intended. !!
            self.migrate_to(idx_r, idx_c, search_empty_cells=search_empty_cells)

    def migrate_to(self, idx_r, idx_c, search_empty_cells=True):
        """ This is the core loop that is intended for NUMBA to vastly speed up the FOR loop.
        Loops through occupied cells, selects a migration location for each cell based on a prob. distribution.
        Reverse search works as follows:
        1 - get populations from all sites a distance d away from the empty site
        2 - calculate carrying capacity (productivity) using env from empty site and skills at occupied sites
        3 - calculate fission distribution for all sites from step 1
        4 - take sites for which fission is True and which are above the migration threshold
        5 - get distribution based on carrying capacities to select site with the best skill / env match
        """
        # how to add flag - we get possible neighbors first based on distance - check if they are empty and habitable
        # don't use migration threshold !!
        # if no empty and habitable cells - set has_valid_neighbors to False for rr, cc
        # For revers search: We search loop through empty cells - no speedup needed

        # Loop through non-zero cells
        for rr, cc in zip(idx_r, idx_c):
            candidates_r = self.indices_r + rr  # get index mask
            candidates_c = self.indices_c + cc

            # Check if within lattice
            within_lattice = (0 <= candidates_r) & (candidates_r < self.shape[1]) & (0 <= candidates_c) & (
                    candidates_c < self.shape[2])
            candidates_r = candidates_r[within_lattice]
            candidates_c = candidates_c[within_lattice]

            # TODO: Ask Daniel how to make this more efficient
            if self.max_distance is not None:
                distances = self._gaez_instance.get_arc_distance((rr, cc), candidates_r, candidates_c)
                mask = distances < self.max_distance
                candidates_r = candidates_r[mask]
                candidates_c = candidates_c[mask]

            if search_empty_cells:
                if self.use_valid_neighs_flag:  # check if there are empty and habitable neighbors
                    is_habitable = np.logical_not(self.is_uninhabitable[candidates_r, candidates_c])
                    is_empty = self.is_empty[candidates_r, candidates_c]
                    # if no empty and habitable neighbors, set has_valid_neighs flag to False
                    # no empty and habitable neighbors - is_empty_and_habitable has only Falsies
                    if not np.any(is_habitable & is_empty):
                        self.has_valid_neighs[rr, cc] = False

                env = self.env[candidates_r, candidates_c]
                skill = self.skills[rr, cc]
                prods = toolbox.calculate_productivity(skill, env,
                                                       prod_scaling=self.max_productivity, min_prod=self.prod_min)

                # Check if productivity of neighboring cells is above thresh and if cells are empty
                mask = (prods >= self.migration_thresh) & self.is_empty[candidates_r, candidates_c]
                prods = prods[mask]
                if prods.size == 0:
                    continue  # continue if no valid neighbors

                if self.search_intelligently:
                    probabilities = toolbox.get_distribution(prods, mn=self.migration_thresh)
                else:
                    probabilities = None

            # TODO: Improve readability of this if / else statement
            else:  # probability distribution to choose which occupied village splits
                populations = self.population[self.num_iter, candidates_r, candidates_c]

                # Get productivity for migration threshold
                env = self.env[rr, cc]  # environment at empty cell
                skill = self.skills[candidates_r, candidates_c]  # skills of villages that will possibly split
                prods = toolbox.calculate_productivity(env, skill,
                                                       prod_scaling=self.max_productivity, min_prod=self.prod_min)

                # Select cells that actually fission, and that have enough skill / env overlap
                prob_to_split = toolbox.get_modulated_distribution(populations,
                                                                   bounds=(self.pop_min, self.pop_max),
                                                                   kind=self.fission_distribution_type)
                floats = self.rng.uniform(low=0.0, high=1.0, size=prob_to_split.shape)
                cells_that_split = prob_to_split > floats
                mask = (prods >= self.migration_thresh) & cells_that_split

                prods = prods[mask]
                if prods.size == 0:
                    continue  # continue if no villages have enough population

                if self.search_intelligently:
                    probabilities = toolbox.get_distribution(prods, mn=self.migration_thresh)
                else:
                    probabilities = None

            candidates_r, candidates_c = candidates_r[mask], candidates_c[mask]  # prob. and cand. have equal dims

            # distribution is either None (uniform) or some nd-array - select accordingly
            idx_select = self.rng.choice(range(candidates_r.size), p=probabilities)
            r0, c0 = candidates_r[idx_select], candidates_c[idx_select]

            if search_empty_cells:  # assign half of the population from (rr, cc) to (r0, c0)
                self.population[self.num_iter, r0, c0] = self.population[self.num_iter, rr, cc] / 2
                self.population[self.num_iter, rr, cc] -= self.population[self.num_iter, rr, cc] / 2
                self.is_empty[r0, c0] = False
                self.skills[r0, c0] = self.skills[rr, cc]  # skills are copied

            else:  # take half of the population from (r0, c0) and assign to (rr, cc)
                num_villagers = min(self.population[self.num_iter, r0, c0] / 2, self.pop_max)
                self.population[self.num_iter, r0, c0] -= num_villagers
                self.population[self.num_iter, rr, cc] = num_villagers
                self.is_empty[rr, cc] = False
                self.skills[rr, cc] = self.skills[r0, c0]

    def run(self, disable_progress_bar=False, track_skill_start=False):
        """Run the simulation for #sim_steps. If mutate_env is not None, at each iteration
        the entries of each binary vector get flipped with prob. mutate_env. Same for mutate_skill.
        If disable_progress_bar is True, do not print tqdm progressbar.
        If track_prod is True, store the productivity per turn"""
        # Check if neighborhood has been set
        if not any(self.indices_r):
            raise ValueError("Run method set_search_params first")

        for _ in tqdm(range(1, self.population.shape[0]), leave=True, disable=disable_progress_bar):
            self.move_forward()
            self.num_iter += 1
