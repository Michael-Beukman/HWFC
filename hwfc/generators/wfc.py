import json
import pickle
import sys


from collections import defaultdict
import datetime
import os
import random
import sys
from typing import Any, Callable, Dict, List, Tuple
from timeit import default_timer as tmr

import numpy as np
from hwfc.utils.classes import InputPattern, Tile
from hwfc.utils.helpers import (
    get_all_tile_options_from_all_patterns,
    get_patterns_from_tilemap,
    load_example_from_file,
    save_pattern_to_txt_file,
    save_world_to_txt_file,
)
from hwfc.utils.mytypes import TOT_TILES
from hwfc.solver import recursive_solve_hierarchy
from hwfc.utils.viz import plot_world_or_pattern
from hwfc.world import World
from hwfc.utils import conf


class WFCGenerator:
    """A standard WaveFunction Collapse generator. This works based on an overlapping pattern model."""

    def __init__(
        self,
        level_size: Tuple[int, int],
        domain: str,
        base_examples: List[str],
        desired_sizes: List[Tuple[int, ...]] = [(3, 3)],
        name: str = "Normal_WFC",
        replace_date_with_seed: bool = False,
        seed: int = None,
        ignore_superpos_tiles: bool = False,
        check_cropped_done: bool = True,
        max_pat_occurrence: int = conf.MAX_PATTERN_OCCURRENCE,
        seed_world_func: Callable[[World], World] = None,
        plot_correct_aspect_ratio: bool = False
    ) -> None:
        """This creates the object.

        Args:
            level_size (Tuple[int, int]): Size of the level, in tiles.
            domain (str): A string name of the domain, e.g., "minecraft".
            base_examples (List[str]): A list of example text files.
            desired_sizes (List[Tuple[int, ...]], optional): A list of sizes for patterns that will be extracted from the examples. Defaults to [(3, 3)].
            name (str, optional): A string indicating the method name. Defaults to "Normal_WFC".
            replace_date_with_seed (bool, optional): If this is true, files will be saved to a file labelled by the seed instead of the date (default). Defaults to False.
            seed (int, optional): If not None, seeds the randomness. Defaults to None.
            ignore_superpos_tiles (bool, optional): If True, ignores patterns that contain superpositions from the input examples. Defaults to False.
            check_cropped_done (bool, optional): If True, terminates generation when the inner level, ignoring the 1-tile thick boundary is fully generated. This is because the boundaries often have problems with satisfying conditions. Defaults to True.
            max_pat_occurrence (int, optional): The maximum count we assign a pattern. Defaults to conf.MAX_PATTERN_OCCURRENCE.
            seed_world_func (Callable[[World], World], optional): A function that takes a world and returns a world. This is used to seed the world with some patterns. Defaults to None.
            plot_correct_aspect_ratio (bool, optional): If True, the plots will be saved with the correct aspect ratio. Defaults to False.
        """
        self.currseed = seed
        self.level_size = level_size
        self.domain = domain
        self.base_examples = base_examples
        self.desired_sizes = desired_sizes
        self.name = name
        self.date_val = self.get_date()
        self.replace_date_with_seed = replace_date_with_seed
        self.ignore_superpos_tiles = ignore_superpos_tiles
        self.dimension = len(level_size)
        self.check_cropped_done = check_cropped_done
        self.max_pat_occurrence = max_pat_occurrence
        self.seed_world_func = seed_world_func
        self.plot_correct_aspect_ratio = plot_correct_aspect_ratio
        self.load_examples()


    def generate(self, seed: int=42) -> World:
        """Generates with the given seed and returns the completed world.

        Args:
            seed (int, optional): Defaults to 42.

        Returns:
            World: 
        """
        _start = tmr()
        self.seed(seed)
        world = World(self.all_tile_options, self.level_size)
        if self.seed_world_func is not None: world = self.seed_world_func(world)
        
        # The function expects this, but we basically do a no-op.
        def hierarchy_cleanup(h_index, curr):
            return curr
        # Generate
        ans = recursive_solve_hierarchy(
            state=world,
            list_of_patterns=[self.pattern_list],
            list_of_max_counts=[1.0],
            list_of_propagate_patterns=[self.only_propagate_pats],
            list_of_should_randomise_coords=[False],
            list_of_hierarchy_cleanups=[hierarchy_cleanup],
            h_index=0,
            strict=True,
            save_loc=self.get_save_loc_for_steps(),
            number_of_hierarchy_levels=1,
            check_cropped_done=self.check_cropped_done,
            desired_sizes=self.desired_sizes,
        )

        _end = tmr()
        self.save_at_end(ans, _end - _start)
        return ans

    ################################# Utils #################################
    def load_examples(self):
        """
            Loads the examples from the files.
        """
        self.pattern_list = []
        self.only_propagate_pats = []
        for ex in self.base_examples:
            
            # Rewrite this if statement in one line:
            example = load_example_from_file(ex) if type(ex) == str else ex

            for size in self.desired_sizes:
                self.pattern_list += get_patterns_from_tilemap(
                    example, size, do_rotate=False, ignore_superpos=self.ignore_superpos_tiles, max_occurrence=self.max_pat_occurrence
                )
            
            self.only_propagate_pats += get_patterns_from_tilemap(
                example, list([2] * self.dimension), do_rotate=False, ignore_superpos=self.ignore_superpos_tiles, max_occurrence=self.max_pat_occurrence
            )
        # Give all patterns an ID.
        self.only_propagate_pats = self.only_propagate_pats
        for i, p in enumerate(self.pattern_list + self.only_propagate_pats):
            p.id = i

        dic_of_counts = defaultdict(lambda: 0)
        for p in self.pattern_list:
            dic_of_counts[p] += p.occurrence

        self.pattern_list = set(self.pattern_list)
        self.only_propagate_pats = list(set(self.only_propagate_pats) - self.pattern_list) # These patterns are the ones we propagate with, but do not collapse with.
        self.pattern_list = list(self.pattern_list)

        for p in self.pattern_list:
            p.occurrence = dic_of_counts[p]

        # Gets all of the options
        self.all_tile_options = get_all_tile_options_from_all_patterns(np.array(self.pattern_list))

        # Saves the patterns
        self.save_patterns(self.pattern_list, "standard")
        self.save_patterns(self.only_propagate_pats, "propagate_patterns")

    def get_date(self) -> str:
        # Returns a prettily formatted date.
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def seed(self, seed: int):
        # Seeds the random generations
        self.currseed = seed
        np.random.seed(seed)
        random.seed(seed)


    ################################# Saving #################################
    
    def already_exists(self) -> bool:
        # Returns True if this has already been generated.
        return os.path.exists(f"{self.get_folder_to_save()}/plots/final_level.png")

    def get_folder_to_save(self) -> str:
        """Returns a string where all the outputs from this level can be saved. It also creates the folder if it does not exist.

        Returns:
            str: 
        """
        D = str(self.currseed).zfill(4) if self.replace_date_with_seed else self.date_val
        if conf.EXTRA_FILE != "":
            d = f"{conf.EXTRA_FILE}/{self.domain}/{self.name}/{D}"
        else:
            d = f"all_outputs/{self.domain}/{self.name}/{D}"
        d1 = f"{d}/plots"
        d2 = f"{d}/texts"
        os.makedirs(d1, exist_ok=True)
        os.makedirs(d2, exist_ok=True)
        return d

    def get_hyperparams(self) -> Dict[str, Any]:
        """Returns a list of hyperparameters of this method.

        Returns:
            Dict[str, Any]: 
        """
        return {
            "seed": self.currseed,
            "level_size": self.level_size,
            "domain": self.domain,
            "base_examples": self.base_examples,
            "desired_sizes": self.desired_sizes,
            "name": self.name,
            "base_examples": self.base_examples,
        }

    def remove_edge(self, map: World) -> World:
        """Modifies the map in place to remove the boundary row and columns.

        Args:
            map (World): 

        Returns:
            World: 
        """
        assert len(map.shape) == 2
        map.shape = (map.shape[0] - 2, map.shape[1] - 2)
        map.map = map.map[1:-1, 1:-1]
        return map


    def save_at_end(self, answer_to_save: World, generation_time: float=0):
        """Saves the hyperparameters and output file to a directory.

        Args:
            answer_to_save (World): 
            generation_time (float, optional): The time it took to generate. Defaults to 0.
        """
        if self.domain == "minecraft":
            filename = f"{self.get_folder_to_save()}/final_map.p"
            with open(filename, 'wb+') as f:
                pickle.dump(answer_to_save, f)
        else:
            if answer_to_save is not None:
                # Save the cropped and non-cropped versions.
                cropped = answer_to_save.get_cropped()
                plot_world_or_pattern(answer_to_save, filename := f"{self.get_folder_to_save()}/plots/final_level.png" , correct_aspect_ratio=self.plot_correct_aspect_ratio)
                plot_world_or_pattern(cropped, filename := f"{self.get_folder_to_save()}/plots/final_level_cropped.png", correct_aspect_ratio=self.plot_correct_aspect_ratio)
                plot_world_or_pattern(
                    cropped,
                    filename := f"{self.get_folder_to_save()}/plots/final_level_cropped_clean.png",
                    make_empty=True, correct_aspect_ratio=self.plot_correct_aspect_ratio
                )
                save_world_to_txt_file(answer_to_save, f"{self.get_folder_to_save()}/texts/final_level.txt")
                save_world_to_txt_file(cropped, f"{self.get_folder_to_save()}/texts/final_level_cropped.txt")
                print(f"Saved Final Level to {filename}")
            else:
                print("Cannot Save Level because it is None")

        hyperparams = self.get_hyperparams()
        hyperparams["generation_time"] = generation_time
        F = f"{self.get_folder_to_save()}"

        with open(f"{F}/hyperparams.json", "w+") as f:
            json.dump(hyperparams, f)

    def save_patterns(self, list_of_pats: List[InputPattern], subfolder: str):
        """Saves the patterns in text files.

        Args:
            list_of_pats (List[InputPattern]): 
            subfolder (str): 
        """
        j = {}
        F = f"{self.get_folder_to_save()}/texts/patterns/{subfolder}"
        os.makedirs(F, exist_ok=True)
        for p in list_of_pats:
            save_pattern_to_txt_file(p, f"{F}/{str(p.id).zfill(4)}.txt")
            j[p.id] = p.occurrence
        with open(f"{F}/occurrences.json", "w+") as f:
            json.dump(j, f)

    def get_save_loc_for_steps(self) -> str:
        # Returns a string where the intermediate steps can be saved
        return f"{self.get_folder_to_save()}/plots/steps/"
