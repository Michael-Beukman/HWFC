import tqdm

from hwfc.generators.wfc import WFCGenerator

import itertools
import sys
from typing import Any, Dict, List, Tuple, Union
from timeit import default_timer as tmr

import numpy as np
from hwfc.utils.classes import Tile
from hwfc.utils.helpers import (
    get_all_tile_options_from_all_patterns,
    get_patterns_from_tilemap,
    load_example_from_file,
)
from hwfc.utils.mytypes import ALL_SUPER_POSITIONS, SUPER_POS, TOT_TILES
from hwfc.solver import recursive_solve_hierarchy
from hwfc.world import World


class HWFCGenerator(WFCGenerator):
    """This inherits from the standard WFC generator, but implements our hierarchical extension."""
    def __init__(
        self,
        level_size: Tuple[int, int],
        domain: str,
        base_examples: List[str],
        hierarchy_examples: List[List[str]],
        desired_sizes=[(3, 3)],
        name: str="Hierarchical_WFC",
        max_counts: List[Union[int, float]]=[1, 0.35, 1.0],
        max_counts_per_patterns: List[Union[List[int], None]]=None,
        **kwargs
    ) -> None:
        """Constructs this object. Most of the parameters are the same as the normal WFC function.

        Args:
            level_size (Tuple[int, int]): Size of the level, in tiles.
            domain (str): A string name of the domain, e.g., "minecraft".
            base_examples (List[str]): A list of example text files.
            hierarchy_examples (List[List[str]]): A list of lists. Each element is a list of example text files, to be used at that level of the hierarchy. For instance, [['top_hierarchy1.txt', 'top_hierarchy2.txt], ['mid_hierarchy1.txt', 'mid_hierarchy2.txt]]. The lowest level is populated from the base example. Do note that each of these patterns should be the hierarchy itself (whereas the base example is used only to extract patterns from).
            desired_sizes (List[Tuple[int, ...]], optional): A list of sizes for patterns that will be extracted from the examples. Defaults to [(3, 3)].
            name (str, optional): A string indicating the method name. Defaults to "Hierarchical_WFC".
            max_counts (List[Union[int, float]], optional): A list indicating when the method should progress to the next hierarchical level. There must be the same number of elements as hierarchy levels (including the lowest one). Each element can either be an integer or a float. Integers N indicate that we should place N elements at that hierarchical level, and a float 0 <= F <= 1 means that we should fill up to F% of the level. Defaults to [1, 0.35, 1.0].
            max_counts_per_patterns (List[List[int]], optional): If given, this has to have the same size as the number of hierarchical levels. Each element must be a list of ints or None. None means that there are no restrictions on that hierarchical level. Otherwise, each integer indicates how many times each pattern at that level must be placed; for instance [[1, 1], None, None] would specify that one of each of the top-level hierarchical (there are 2) must be placed.  Defaults to None.
        """
        self.hierarchy_examples = hierarchy_examples
        self.max_counts = max_counts
        self.max_counts_per_patterns = max_counts_per_patterns
        self.dict_mapping = None
        super().__init__(level_size, domain, base_examples, desired_sizes, name=name, **kwargs)

    def generate(self, seed=42) -> World:
        self.seed(seed)
        _start = tmr()
        world = World(self.all_tile_options, self.level_size)
        world.dict_mapping = self.dict_mapping
        if self.seed_world_func is not None: world = self.seed_world_func(world)

        curr = world
        max_counts = self.max_counts
        all_hierarchies = self.hierarchy_levels + [self.pattern_list]

        def hierarchy_cleanup(h_index: int, curr: World):
            # This is the cleanup step after each hierarchical level
            
            
            # If we are at the top of the hierarchy, we need to change all uncollapsed tiles to -2, because nothing can be placed there at this level.
            # Here one could change the behaviour to place medium-level hierarchies *outside* of the top-level hierarchies.
            ranges_to_iterate_over = [list(range(0, 0 + s)) for s in curr.shape]
            if h_index == 0:
                # make all uncollapsed tiles -2
                for coord in itertools.product(*ranges_to_iterate_over):
                    col = curr.map[coord]
                    if len(col.possible_tiles) > 1: col.set([ALL_SUPER_POSITIONS[0]], True)

                for coord in itertools.product(*ranges_to_iterate_over):
                    col = curr.map[coord]
                    if col.possible_tiles == [ALL_SUPER_POSITIONS[0]]: col.set([SUPER_POS], True)
                    elif col.possible_tiles == [SUPER_POS]: col.set([ALL_SUPER_POSITIONS[0]], True)

            else:
                curr._put_superpositions_back()
            curr._put_superpositions_back(ALL_SUPER_POSITIONS[0])

            # Now we need to redo the propagation step here to ensure that the new tiles we put in are indeed feasible.
            return self.propagate_after_hierarchy_level(ranges_to_iterate_over, curr, h_patterns, all_propagate_pats)

        # Outside of the hierarchy function, we now call the main code.
        n_hierarchical_levels = len(all_hierarchies)
        list_of_propagate_patterns = []
        list_of_cleanups = [hierarchy_cleanup for _ in range(n_hierarchical_levels)]
        list_of_should_randomise = [True] + [False] * n_hierarchical_levels
        for h_index, h_patterns in enumerate(all_hierarchies):
            # Propagate patterns are all patterns from the current level onwards.
            all_propagate_pats = (
                sum(self.hierarchy_levels[h_index + 1 :] + [self.pattern_list], []) + self.only_propagate_pats
            )
            list_of_propagate_patterns.append(all_propagate_pats)

        # Call the function
        curr = recursive_solve_hierarchy(
            curr,
            all_hierarchies,
            max_counts,
            list_of_propagate_patterns=list_of_propagate_patterns,
            list_of_should_randomise_coords=list_of_should_randomise,
            list_of_hierarchy_cleanups=list_of_cleanups,
            h_index=0,
            strict=True,
            save_loc=self.get_save_loc_for_steps(),
            number_of_hierarchy_levels=len(all_hierarchies),
            check_cropped_done=self.check_cropped_done,
            desired_sizes=self.desired_sizes,
        )
        _end = tmr()
        self.save_at_end(curr, _end - _start)
        return curr

    def load_examples(self):
        """
            Loads the examples. Here we additionally load the hierarchical examples, and extract propagation patterns from them.
        """
        super().load_examples()
        # Load each pattern from a file
        self.hierarchy_levels = [[load_example_from_file(f) for f in h_level] for h_level in self.hierarchy_examples]
        
        # If we have a maximum number of patterns to place, set that here.
        if self.max_counts_per_patterns is not None:
            already_added = set()
            self.dict_mapping = {}
            for h_level, p in zip(self.hierarchy_levels, self.max_counts_per_patterns):
                j = 0
                for r in h_level:
                    if r.global_id in already_added: continue
                    already_added.add(r.global_id)
                    self.dict_mapping[r.global_id] = p[j]
                    j += 1
        
        # Propagate patterns have at least 2x2 in them, in addition to the desired sizes.
        tmp = []
        for size in set([tuple([2] * self.dimension)] + self.desired_sizes):
            for h_level in self.hierarchy_levels:
                for ex in h_level:
                    tmp += get_patterns_from_tilemap(ex, size, do_rotate=False, max_occurrence=self.max_pat_occurrence)
        for i, p in enumerate(tmp):
            p.id = i + len(self.only_propagate_pats)
        self.only_propagate_pats += tmp

        # Here we make sure these patterns are not duplicated.
        self.only_propagate_pats = list(set(self.only_propagate_pats) - set(self.pattern_list))

        self.save_patterns(self.only_propagate_pats, "hierarchy_propagate_pats")

        if self.domain == "minecraft":
            # Need to update here to be consistent
            self.all_tile_options = get_all_tile_options_from_all_patterns(
                np.array(self.only_propagate_pats + self.pattern_list)
            )

    def get_hyperparams(self) -> Dict[str, Any]:
        a = super().get_hyperparams()

        return a | {
            "hierarchy_examples": self.hierarchy_examples,
            "max_counts": self.max_counts,
        }

    def propagate_after_hierarchy_level(self, ranges_to_iterate_over: List[Tuple[int, ...]], curr: World, h_patterns, all_propagate_pats):
            all_tiles_to_consider_for_propagation = set()
            for coord in itertools.product(*ranges_to_iterate_over):
                sizes = [list(range(r)) for r in tuple([3] * len(curr.shape))] # A small radius
                # If this is collapsed, and it is not a superposition, then we need to propagate its effect to the recently uncollapsed tiles
                if curr.map[coord].is_collapsed and curr.map[coord].possible_tiles[0] not in {SUPER_POS} | set(
                    ALL_SUPER_POSITIONS
                ):
                    for tmp in itertools.product(*sizes):
                        T = np.array(tmp) + np.array(coord) # add the center (coord) to the relative coordinate (tmp)
                        all_tiles_to_consider_for_propagation.add(tuple(T))

            T = (True, 1)
            prev_num = -1
            prev_count = 0
            while T != False and T[1] > 0:
                queue_override = list(all_tiles_to_consider_for_propagation)
                T = curr.propagate_constraints(
                        h_patterns + all_propagate_pats,
                        tuple([1] * len(curr.shape)),
                        coord,
                        be_strict=True,
                        return_counts=True,
                        queue_override=queue_override
                    )
                if T != False:
                    if prev_num == T[1]:
                        prev_count += 1
                    else:
                        prev_count = 0
                        prev_num = T[1]
                if prev_count > 5:
                    break
            return curr
