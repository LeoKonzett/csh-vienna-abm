import numpy as np
import noise
from tqdm import tqdm
import logging
import toolbox


# TODO: General. I have too many small functions that are used only once. We only need functions if we re-use.

class Lattice:
    """Base object for population dynamics in medieval Europe. Square lattice.
    Central object is a square lattice with shape (time, x, x) which keeps the track of the population in each village.
    """
    _gaez_instance = None  # instance of GlobalAezV4 in gaez_data_loader.py

    def __init__(self, steps_sim=5_000, seed=50, idx_start=(0, 0), num_env_vars=10,
                 pop_min=100, pop_max=500, rate_growth=1 / 30, prod_min=1, rate_prod=200,
                 n_rows=10, n_cols=10, env=None):
        """ Initializes lattice object with shape (time=steps_sim, n_rows, n_cols). assumes that env. vector
        and skill vector have the same length. Also initializes environment and skill vectors to calculate productivity.
        Current data type of population array is float (integer messes up the update rule)
        """
        # TODO: Refactor checks to improve readability
        self.rng = np.random.default_rng(seed=seed)
        self.r0, self.c0 = idx_start  # location of first settlement
        self.shape = (steps_sim, n_rows, n_cols)  # only square lattice - time is 0 axis
        self.population = np.zeros(self.shape)  # TODO: Two arrays A) Total population 1D and 2D population per step
        self.population[0, self.r0, self.c0] = 10  # 10 people initially

        # min / max population for a village to split (i.e. split with p=1 above max)
        self.pop_min, self.pop_max = pop_min, pop_max

        self.rate_growth = rate_growth  # set growth of population rate
        self.rate_prod = rate_prod  # If this is large, small changes in the productivity are amplified

        # Deal with negative population growth:
        thresh = self.pop_max / (2 * (1 + 1 / self.rate_growth))
        self.prod_min = prod_min
        assert prod_min >= 0 and isinstance(prod_min, int), "Provide minimum productivity as positive integer"
        if prod_min <= thresh:
            print("In Lattice.init().: To avoid negative population, clip population to zero.")
            self.clip_population_to_zero = True
        else:
            self.clip_population_to_zero = False

        # load environment
        if env is None:
            self.num_env_vars = num_env_vars
            self.env = self.init_env_perlin(scale=0.2)  # perlin env with scale 0.2 for now
        else:
            assert np.all(env.shape == self.shape[1:]), f"Input with shape {env.shape} doesn't match " \
                                                        f"shape {self.shape[1:]}."
            self.init_env_from_gaez(env)  # loads self.env

        print(f"Cell productivity varies in steps {int(self.rate_prod / self.num_env_vars)}. In case of GAEZ env., "
              f"this is also the maximum productivity.")
        assert self.rate_prod / self.num_env_vars > self.pop_max, "With current parameters, " \
                                                                  "village will not split. Increase either" \
                                                                  "productivity base rate or reduce " \
                                                                  "population required to split."

        # initialize skills - assign random bin. vector with length (num_skill_vars) to starting village
        self.num_skill_vars = self.num_env_vars
        self.skills = np.zeros_like(self.env)
        self.skills[self.r0, self.c0] = self.rng.integers(size=self.num_skill_vars, low=0, high=1, endpoint=True)
        self.prod = np.zeros((n_rows, n_cols))
        self.prod = toolbox.calculate_productivity(self.skills, self.env,
                                                   prod_scaling=self.rate_prod, min_prod=self.prod_min)

        self.num_iter = 1
        self.is_empty = self.population[0, :, :] == 0  # flag empty cells

        # TODO: Add reminder that these MUST be initialized or give warning if they aren't
        self.indices_r, self.indices_c = [], []  # indices to select neighbors
        self.prod_threshold = 0  # min. productivity for villagers to migrate
        self.search_intelligently = False  # If True, select cells probabilistically based on productivity

        self.env_mutation_rate = None  # Probability that one entry of the environment vector per site flips
        self.skill_mutation_rate = None  # Same but for skills
        self.mutation_method = None  # True is random mutation, False is Metropolis mutation
        self.max_distance = None  # maximum distance between two points in km

        self.repopulate_empty_cells = False  # Repopulate dead villages - don't if no mutations for skill and env.
        self.metropolis_scale = 1  # scaling factor for metropolis algorithm

    def init_env_from_gaez(self, input_arr, water_var=0):
        """ load environment based on gaez v4 data set (33 AEZ classes, 5 arc-minute resolution)
        input data is an integer array with entries {0, 33}, where 0 is water (water_var) and 32 is built-up land.
        precise docs can be found at Gaez V4 user guide, page 162.
        output data is a 3D array with exactly one non-zero entry along the last axis that denotes
        the AEZ class to which the village belongs.
        """
        variables = np.unique(input_arr)  # different AEZ classes
        self.num_env_vars = variables.size

        self.env = np.zeros([*input_arr.shape, self.num_env_vars], dtype=float)
        for idx, val in enumerate(variables):  # start from 1 to skip water entries
            mask = input_arr == val
            self.env[mask, idx] = 1

        # if successful, sum along last axis is unity
        assert np.all(np.sum(self.env, axis=-1) == 1), f"sum is non-unity and is {np.sum(self.env, axis=-1)}"

        # handle water - can be extended to other variables
        if water_var in variables:
            print("In Lattice.init_env_from_gaez(): Input environment contains water. Set to zero. "
                  "Ensure that productivity settlement threshold is above 1 to avoid settling water.")
            is_water = input_arr == water_var
            water_idx = np.squeeze(np.argwhere(variables == water_var))
            self.env[is_water, water_idx] = 0

            if is_water[self.r0, self.c0]:  # starting location in water
                raise Exception("Staring point is in water. Choose different point.")

    def init_env_perlin(self, scale=0.1):
        """Create an environment of shape (size, size, num_entries) with Perlin noise.
        This yields spatially correlated noise. scale gives the correlation length"""
        # Populate environment with spatially correlated variables
        env_perlin = np.zeros((*self.shape[1:], self.num_env_vars))
        for ii in range(self.shape[1]):  # rows
            for jj in range(self.shape[2]):  # columns
                env_perlin[ii, jj] = [noise.pnoise3(ii * scale, jj * scale, k * scale) for k in
                                      range(self.num_env_vars)]

        # Binarize the array
        for ii in range(self.num_env_vars):
            env_perlin[:, :, ii] = np.where(env_perlin[:, :, ii] > 0, 1, 0)

        return env_perlin

    def load_gaez_instance(self, gaez_instance):
        """load an instance of Global_AEZ data. Is pass-by-reference"""
        self._gaez_instance = gaez_instance
        # TODO: Don't access private variables
        assert gaez_instance._nrows == self.shape[1] and gaez_instance._ncols == self.shape[2]

    def get_split_probs(self, population):
        """ Calculate the probability that each village splits. Uses uniform cdf."""
        factor = 1 / (self.pop_max - self.pop_min)
        raw = factor * (population - self.pop_min)
        return np.clip(raw, 0, 1)

    def set_search_params(self, prod_threshold=100, neigh_type="von_neumann", distance=1,
                          search_intelligently=False, max_distance_km=None):
        """ Sets the type of search environment. For now: Lattice with Moore and VN neighborhoods.
        If the productivity of the selected cell is below prod_threshold, the cell doesn't split.
        distance attribute sets the size of the neighborhoods (e.g. distance 1 for Moore is 3x3)
        """
        self.indices_r, self.indices_c = toolbox.set_neighborhood(distance=distance, neigh_type=neigh_type)
        self.prod_threshold = prod_threshold
        assert self.prod_threshold > 0, "Current version requires a settlement threshold > 0"
        self.search_intelligently = search_intelligently
        if max_distance_km is not None:
            self.max_distance = max_distance_km
            print(f"Villagers have maximum search radius of {max_distance_km} kilometres.")

    def set_evolution_params(self, env_mutation_rate=None, skill_mutation_rate=None, mutation_method="metropolis",
                             metropolis_scale=1, repopulate_empty_cells=False):
        """Sets / updates parameters for evolution, e.g. environment / skill mutation rate or prod. scaling factor.
        For now, only updates the mutation rates."""
        self.env_mutation_rate = env_mutation_rate  # environment mutation rate
        self.skill_mutation_rate = skill_mutation_rate  # skill mutation rate
        self.mutation_method = mutation_method  # mutate directionally (metropolis) or random
        self.metropolis_scale = metropolis_scale  # scaling ratio for Metropolis
        self.repopulate_empty_cells = repopulate_empty_cells  # repopulate dead villages

        # Checks
        if env_mutation_rate is None and skill_mutation_rate is None and self.repopulate_empty_cells:
            logging.warning("If neither the environment nor the skill mutates, it is "
                            "strongly advised to set repopulate_empty_cells to False to speed up the code.")

        if self.clip_population_to_zero and not self.repopulate_empty_cells:
            logging.warning("Villages can die (zero population) but cannot be re-populated.")

    def mutate_skill_metropolis(self, p_flip, mask_additional=None, scale=1):
        """Metropolis-like mutation for skills. Compares the ratio alpha = Prod(flip) / prod_previous and accepts
        the proposed flip if alpha = num where num is a uniform float in [0, 1]
        If scale != 1, the ratio array gets exponentiated by scale, e.g. ratio -> ratio ** scale
        This exacerbates larger differences, e.g. 0.75 -> 0.56 (scale = 2) -> 0.42
        """
        prod_previous = toolbox.calculate_productivity(self.skills, self.env, prod_scaling=self.rate_prod,
                                                       min_prod=self.prod_min)

        skills_update = self.flip_single_entry_per_cell(self.skills, p_flip=p_flip, mask_additional=mask_additional)
        prod_update = toolbox.calculate_productivity(skills_update, self.env, prod_scaling=self.rate_prod,
                                                     min_prod=self.prod_min)
        with np.errstate(divide="ignore", invalid="ignore"):  # ignore division by zero warning
            ratio = np.divide(prod_update, prod_previous)

        ratio = np.power(ratio, scale)  # apply custom scaling

        nums = self.rng.uniform(low=0, high=1., size=self.prod.shape)
        accept_flip = ratio > nums  # accept flip dep. on ratio

        self.skills[accept_flip] = skills_update[accept_flip]

        return self.skills

    def mutate_skill_diff_ratios(self, probabilities=None, mutation_rate=None):
        """Skill mutation happens with rate r. We identify 4 different cases:
        Gain of useful skill. Loss of useful skill. Gain of useless skill.
        Loss of useless skill. We assign different probabilities for all 4 cases."""
        if probabilities is None:  # uniform distribution
            probabilities = [0.25] * 4
        else:
            assert len(probabilities) == 4, "Provide 4 probability values."

        p_gain_useful, p_gain_useless, p_lose_useful, p_lose_useless = probabilities

        nums = self.rng.uniform(low=0, high=1, size=self.prod.shape)
        mask_2d = nums > mutation_rate

        # get r/c indices of cells for which a flip happens
        idx_r, idx_c = np.nonzero(mask_2d)

        # pick a random number for each flipped cell
        idx_flip = self.rng.choice(range(self.num_env_vars), size=idx_r.size)

        # extract relevant values
        env_vals = self.env[idx_r, idx_c, idx_flip]
        skill_vals = self.skills[idx_r, idx_c, idx_flip]

        # get acceptance probabilities
        m1 = (skill_vals == 0 & env_vals == 0)  # gain skill that isn't used
        m2 = (skill_vals == 0 & env_vals == 1)  # gain skill that is used
        m3 = (skill_vals == 1 & env_vals == 0)  # lose skill that isn't used

        probabilities = np.select([m1, m2, m3], [p_gain_useless, p_gain_useful, p_lose_useless], default=p_lose_useful)
        nums = self.rng.uniform(low=0, high=1, size=env_vals.size)
        mask_1d = nums > probabilities
        
        # enable accepted flips and replace
        replacement_vals = np.logical_xor(skill_vals, mask_1d).astype(int)
        self.skills[mask_2d] = replacement_vals

    def flip_single_entry_per_cell(self, array, p_flip=0.01, mask_additional=None):
        """Flip one randomly selected entry (uniform pdf with rng) with probability p_flip.
        Env has shape (N, N, 10) - we need NxN random floats in [0, 1].
        E.g. for a NxN lattice, we have one expected flip per function call if p_flip = 1 / (NxN)
        If mask is not None, flip only where Mask is True (in addition to random draw)
        """
        nums = self.rng.uniform(low=0, high=1, size=self.prod.shape)
        mask = nums < p_flip  # select cells for which a flip happens

        if mask_additional is not None:
            print("In Lattice.flip_single_entry_per_cell: mask_additional not None. Deprecated.")
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

        if self.skill_mutation_rate is not None:
            if self.mutation_method == "random":
                self.skills = self.flip_single_entry_per_cell(self.skills, p_flip=self.skill_mutation_rate,
                                                              mask_additional=None)
            elif self.mutation_method == "metropolis":
                self.skills = self.mutate_skill_metropolis(p_flip=self.skill_mutation_rate,
                                                           mask_additional=None, scale=10)
                
            elif self.mutation_method == "4_rates":  # TODO: Ordering in probabilities is important - FIX
                self.mutate_skill_diff_ratios(mutation_rate=self.skill_mutation_rate,
                                              probabilities=(0.3, 0.1, 0.2, 0.9))  # u g, non-u g, u l, non-u l

        # calculate productivity to update village population
        self.prod = toolbox.calculate_productivity(self.skills, self.env,
                                                   prod_scaling=self.rate_prod, min_prod=self.prod_min)

        # get empty cells from previous iteration
        self.is_empty = self.population[self.num_iter - 1] == 0

        # update population - empty cells will stay empty regardless of productivity
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.population[self.num_iter - 1, self.r0, self.c0] < 1:
                self.population[self.num_iter - 1, self.r0, self.c0] = 1  # ensure one villager remains

            # TODO: Could be optimized for memory usage but Lattice.migrate_to() takes more run-time
            self.population[self.num_iter] = self.population[self.num_iter - 1] + \
                self.rate_growth * self.population[self.num_iter - 1] \
                * (1 - np.divide(self.population[self.num_iter - 1], self.prod))

        if self.clip_population_to_zero:  # set NaN or negative populations to zero
            mask = np.isnan(self.population[self.num_iter]) | (self.population[self.num_iter] < 0)
            self.population[self.num_iter][mask] = 0

        # Get split probability - based on global float draw
        prob_to_split = self.get_split_probs(self.population[self.num_iter])
        floats = self.rng.uniform(low=0.0, high=1.0, size=self.prod.shape)

        # Count the number of cells that want to split and set helper variable
        num_cells_splitting = np.sum(prob_to_split > floats)
        if num_cells_splitting <= (np.prod(self.prod.shape) / 2):
            search_empty_cells = True
        else:
            search_empty_cells = False

        cells_that_split = prob_to_split > floats
        if search_empty_cells:  # villagers that move to empty locations
            idx_r, idx_c = np.nonzero(cells_that_split)
        else:  # reverse search
            idx_r, idx_c = np.nonzero(self.is_empty & (self.prod > self.prod_threshold))

        if idx_r.size > 0:
            self.migrate_to(idx_r, idx_c, search_empty_cells=search_empty_cells)

    def migrate_to(self, idx_r, idx_c, search_empty_cells=True):
        """ This is the core loop that is intended for NUMBA to vastly speed up the FOR loop.
        Loops through occupied cells, selects a migration location for each cell based on a prob. distribution."""

        # Loop through non-zero cells
        for rr, cc in zip(idx_r, idx_c):
            candidates_r = self.indices_r + rr  # get index mask
            candidates_c = self.indices_c + cc

            # Check if within lattice # TODO: Get directly from memory to avoid check? Speed-up?
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

            # TODO: Also picks villages below the productivity threshold
            if not self.search_intelligently:  # pick village to migrate to at random
                probabilities = None  # we pick one candidate regardless of empty or occupied villages

            else:  # use intelligent search strategy
                if search_empty_cells:  # probability distribution for empty villages based on productivity

                    env = self.env[candidates_r, candidates_c]
                    skill = self.skills[rr, cc]
                    prods = toolbox.calculate_productivity(skill, env,
                                                           prod_scaling=self.rate_prod, min_prod=self.prod_min)

                    # Check if productivity of neighboring cells is above thresh and if cells are empty
                    mask = (prods > self.prod_threshold) & self.is_empty[candidates_r, candidates_c]
                    prods = prods[mask]
                    if prods.size == 0:
                        continue  # continue if no valid neighbors

                    probabilities = toolbox.get_distribution(prods, mn=self.prod_threshold)

                else:  # probability distribution to choose which occupied village splits
                    populations = self.population[self.num_iter, candidates_r, candidates_c]
                    mask = populations > self.pop_min
                    populations = populations[mask]
                    if populations.size == 0:
                        continue  # continue if no villages have enough population

                    probabilities = toolbox.get_distribution(populations, mn=self.pop_min)

                candidates_r, candidates_c = candidates_r[mask], candidates_c[mask]  # prob. and cand. have equal dims

            # distribution is either None (uniform) or some nd-array - select accordingly
            idx_select = self.rng.choice(range(candidates_r.size), p=probabilities)
            r0, c0 = candidates_r[idx_select], candidates_c[idx_select]

            # TODO: Rewrite using temporary variable, but readability suffers
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

    def run(self, disable_progress_bar=False):
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
