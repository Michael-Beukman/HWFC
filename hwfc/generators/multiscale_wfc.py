import tqdm

from hwfc.generators.hierarchical_wfc import HWFCGenerator

import itertools
from typing import List, Tuple

import numpy as np
from hwfc.utils.mytypes import ALL_SUPER_POSITIONS, SUPER_POS
from hwfc.solver import recursive_solve_hierarchy
from hwfc.world import World


class MultScaleWFCGenerator(HWFCGenerator):
    """A Multiscale WFC Generator. This naively considers the larger patterns the same as normal patterns and places them where it can.

    Args:
        HWFCGenerator (_type_): _description_
    """

    def __init__(
        self,
        level_size: Tuple[int, int],
        domain: str,
        base_examples: List[str],
        hierarchy_examples: List[List[str]],
        desired_sizes=[(3, 3)],
        name="Multiscale_WFC",
        **kwargs
    ) -> None:
        """These arguments are the same as the HWFC class, except that the `max_counts` argument is not used.

        Args:
            level_size (Tuple[int, int]): Size of the level, in tiles.
            domain (str): A string name of the domain, e.g., "minecraft".
            base_examples (List[str]): A list of example text files.
            hierarchy_examples (List[List[str]]): A list of lists. Each element is a list of example text files, to be used at that level of the hierarchy. For instance, [['top_hierarchy1.txt', 'top_hierarchy2.txt], ['mid_hierarchy1.txt', 'mid_hierarchy2.txt]]. The lowest level is populated from the base example. Do note that each of these patterns should be the hierarchy itself (whereas the base example is used only to extract patterns from).
            desired_sizes (List[Tuple[int, ...]], optional): A list of sizes for patterns that will be extracted from the examples. Defaults to [(3, 3)].
            name (str, optional): A string indicating the method name. Defaults to "Multiscale_WFC".
        """
        super().__init__(level_size, domain, base_examples, hierarchy_examples, desired_sizes, name, [1.0], **kwargs)

    def generate(self, seed=42):
        """This generates the world."""
        self.seed(seed)
        world = World(self.all_tile_options, self.level_size)
        if self.seed_world_func is not None: world = self.seed_world_func(world)

        def hierarchy_cleanup(h_index, curr):
            return curr

        def A(world, cleanup, pats):
            # Generate using only one hierarchical level, with all of the patterns.

            curr = recursive_solve_hierarchy(
                world,
                [pats],
                [1.0],
                list_of_propagate_patterns=[self.only_propagate_pats],
                list_of_should_randomise_coords=[False],
                list_of_hierarchy_cleanups=[cleanup],
                h_index=0,
                strict=True,
                save_loc=self.get_save_loc_for_steps(),
                number_of_hierarchy_levels=1,
                check_cropped_done=True,
                desired_sizes=self.desired_sizes,
                force_random_tilesizes=True,
            )
            return curr

        ranges_to_iterate_over = [list(range(0, 0 + s)) for s in world.shape]
        alls = sum(self.hierarchy_levels + [self.pattern_list], [])
        alls_not_top = sum(self.hierarchy_levels[1:] + [self.pattern_list], [])

        curr = A(world, hierarchy_cleanup, alls)
        curr._put_superpositions_back()


        # Put back all of the superpositions
        curr = self.propagate_after_hierarchy_level(ranges_to_iterate_over, curr, alls, self.only_propagate_pats)
        
        # Here, run it again to fill up the whole level.
        curr = A(curr, hierarchy_cleanup, alls_not_top)
        self.save_at_end(curr)

        return curr
