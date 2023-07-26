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

    def __init__(self, steps_sim=5_000, seed=50, idx_start=(0, 0), num_env_vars=10,
                 pop_min=100, pop_max=500, rate_growth=1 / 30, prod_min=1, rate_prod=200,
                 n_rows=10, n_cols=10, env=None):
        """ Initializes lattice object with shape (time=steps_sim, n_rows, n_cols). assumes that env. vector
        and skill vector have the same length. Also initializes environment and skill vectors to calculate productivity.
        Current data type of population array is float (integer messes up the update rule)
        """
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

        # load environment # TODO: Refactor
        if env is None:
            self.num_env_vars = num_env_vars
            self.env = self.init_env_perlin(scale=0.2)  # perlin env with scale 0.2 for now
        else:
            assert np.all(env.shape == self.shape[1:]), f"Input with shape {env.shape} doesn't match " \
                                                        f"shape {self.shape[1:]}."
            self.init_env_from_gaez(env)  # loads self.env

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
        self.skill_mutates_randomly = True  # True is random mutation, False is Metropolis mutation

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
            print("In meth. init_env_from_gaez(): Input environment contains water. Set to zero."
                  "Ensure that productivity settlement threshold is above 1 to avoid settling water.")
            is_water = input_arr == water_var
            water_idx = np.squeeze(np.argwhere(variables == water_var))
            self.env[is_water, water_idx] = 0

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

    def get_split_probs(self, population):
        """ Calculate the probability that each village splits. Uses uniform cdf."""
        factor = 1 / (self.pop_max - self.pop_min)
        raw = factor * (population - self.pop_min)
        return np.clip(raw, 0, 1)

    def update_population(self, pop_current):
        """Updates the population according to discrete solution of logistic ODE.
        If prod << 1, then the population will turn negative if non-zero (as e.g. after a split).
        To avoid this, we clip the population to zero.
        To avoid division by zero error, we add an epsilon to the divisor.
        Alternatively, we can clip to e.g. -100 to distinguish "dead" from empty villages.
        NB: np.divide(x, 0) with x > 0 returns np.inf. np.divide(0, 0) returns np.nan
        (happens e.g. if population and productivity are both zero)
        Problem: If cell has been populated, but the productivity is zero, then the population will go to zero
        and in the NEXT iteration we will again have NAN. To avoid that, we check is_empty attribute
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            pop_new = pop_current + self.rate_growth * pop_current * (1 - np.divide(pop_current, self.prod))
        assert pop_new.dtype == self.population.dtype

        if np.any(pop_new - pop_current < 1) and np.issubdtype(self.population.dtype, np.integer):
            logging.warning("In meth. update_population: Increase in population of one village is less than unity.")

        if self.clip_population_to_zero:
            np.clip(pop_new, 0, None, out=pop_new)
            mask = np.isnan(pop_new) & np.logical_not(self.is_empty)  # check if village has been occupied and is NaN
            pop_new[mask] = 0

        # These checks are for safety - add way to run this in verbose mode?
        if np.any(pop_new < 0):
            raise ValueError("Population is negative. Either increase minimum productivity or set it to None to"
                             "clip array")

        if np.any(np.isnan(pop_new) & np.logical_not(self.is_empty)):
            raise ValueError("A village that has been populated has a population of NAN instead of zero")

        return pop_new

    def set_search_params(self, prod_threshold=100, neigh_type="von_neumann", distance=1, search_intelligently=False):
        """ Sets the type of search environment. For now: Lattice with Moore and VN neighborhoods.
        If the productivity of the selected cell is below prod_threshold, the cell doesn't split.
        distance attribute sets the size of the neighborhoods (e.g. distance 1 for Moore is 3x3)
        """
        self.indices_r, self.indices_c = toolbox.set_neighborhood(distance=distance, neigh_type=neigh_type)
        self.prod_threshold = prod_threshold
        self.search_intelligently = search_intelligently

    def set_evolution_params(self, env_mutation_rate=None, skill_mutation_rate=None, skill_mutates_randomly=True,
                             metropolis_scale=1, repopulate_empty_cells=False):
        """Sets / updates parameters for evolution, e.g. environment / skill mutation rate or prod. scaling factor.
        For now, only updates the mutation rates."""
        self.env_mutation_rate = env_mutation_rate  # environment mutation rate
        self.skill_mutation_rate = skill_mutation_rate  # skill mutation rate
        self.skill_mutates_randomly = skill_mutates_randomly  # mutate directionally (metropolis) or random
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

        # replace the values - modifies the array in place
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
        # calculate productivity to update village population
        self.prod = toolbox.calculate_productivity(self.skills, self.env,
                                                   prod_scaling=self.rate_prod, min_prod=self.prod_min)
        self.population[self.num_iter] = self.update_population(self.population[self.num_iter - 1])

        # Get split probability - based on global float draw
        prob_to_split = self.get_split_probs(self.population[self.num_iter])
        floats = self.rng.uniform(low=0.0, high=1.0, size=self.prod.shape)

        # Count the number of cells that want to split and set helper variable
        num_cells_splitting = np.sum(prob_to_split > floats)
        if num_cells_splitting <= (np.prod(self.prod.shape) / 2):
            search_empty_cells = True
        else:
            search_empty_cells = False

        self.is_empty = self.population[self.num_iter] == 0
        cells_that_split = prob_to_split > floats

        if search_empty_cells:  # villagers that move to empty locations
            idx_r, idx_c = np.nonzero(cells_that_split)
        else:  # reverse search
            idx_r, idx_c = np.nonzero(self.is_empty & (self.prod > self.prod_threshold))

        if idx_r.size > 0:
            self.migrate_to(idx_r, idx_c, search_empty_cells=search_empty_cells)

        # run mutations
        if self.env_mutation_rate is not None:
            self.env = self.flip_single_entry_per_cell(self.env, p_flip=self.env_mutation_rate)

        if self.skill_mutation_rate is not None:
            is_occupied = np.logical_not(self.is_empty)  # migrate_to updates is_empty flags
            if self.skill_mutates_randomly:
                self.skills = self.flip_single_entry_per_cell(self.skills, p_flip=self.skill_mutation_rate,
                                                              mask_additional=is_occupied)
            else:
                self.skills = self.mutate_skill_metropolis(p_flip=self.skill_mutation_rate,
                                                           mask_additional=is_occupied, scale=10)

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
