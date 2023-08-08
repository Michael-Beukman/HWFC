import json
import random
import sys
from typing import Callable, List, Tuple, Union
import gc
import numpy as np

from hwfc.utils.classes import InputPattern
from hwfc.utils.mytypes import SUPER_POS, Coordinate
from hwfc.utils.viz import plot_world_or_pattern
from hwfc.world import FilterCoord, SuperPosition, World, default_filter_coord
from hwfc.utils import helpers
from hwfc.utils import conf
import os
import tqdm
sys.setrecursionlimit(100_000)


failed_steps = 0
total_counts = 0
RANDOM_PERCENT = 0.8
DEBUG = False
MAX_STEPS = 2_000_000
DO_LOG = False
P_BAR = tqdm.tqdm(total=1.0)

################################# Main Function #################################

def recursive_solve_hierarchy(
    state: World,
    list_of_patterns: List[List[InputPattern]],
    list_of_max_counts: List[Union[int, float]] = None,
    list_of_should_randomise_coords: List[bool] = False,
    h_index: int = 0,
    strict: bool = True,
    list_of_propagate_patterns: List[List[InputPattern]] = [],
    list_of_hierarchy_cleanups: List[Callable[[int, World], World]] = [],
    save_loc: str = "plots/steps",
    number_of_hierarchy_levels: int = 2,
    check_cropped_done: bool = False,
    coord_filter: FilterCoord = default_filter_coord,
    num_recursive_calls: int = 0,
    desired_sizes: List[Coordinate] = [(2, 2), (3, 3)],
    force_random_tilesizes: bool = False,
    recursive_calls_at_start_of_level: int = 0,
) -> World:
    """This is the main (H)WFC implementation. It takes in a current world state, and recursively calls itself until the level is generated, or it fails.

        Generally, many of the below arguments are lists of <thing> This is so that there is a <thing> for each level of the hierarchy.

    Args:
        state (World): The starting world state.
        list_of_patterns (List[List[InputPattern]]): The list of patterns to use at each level of the hierarchy.
        list_of_max_counts (List[Union[int, float]], optional): If given, the `max_count` for each level of the hierarchy. Determines when the method goes to the next level. See the HWFC class for more. Defaults to None.
        list_of_should_randomise_coords (List[bool], optional): A list of booleans, indicating if coordinates should be randomised or collapsed in order of increasing entropy. Defaults to False.
        h_index (int, optional): The current hierarchy index. Defaults to 0.
        strict (bool, optional): Should generally be true, governs when we should fail vs trying again when at a particular level, we fail. Defaults to True.
        list_of_propagate_patterns (List[List[InputPattern]], optional): A list of patterns to use only for propagation and not collapsing. Defaults to [].
        list_of_hierarchy_cleanups (List[Callable[[int, World], World]], optional): A list of functions that take in the current hierarchy level, and world and must return a new world. These are called between the different hierarchy levels. Defaults to [].
        save_loc (str, optional): A location to save the levels at. Defaults to "plots/steps".
        number_of_hierarchy_levels (int, optional): How many total hierarchical levels there are. Defaults to 2.
        check_cropped_done (bool, optional): If true, ends generation when the cropped world is done, otherwise waits until the entire world is done. Defaults to False.
        coord_filter (FilterCoord, optional): This generates only where this coord filter returns true. Defaults to default_filter_coord.
        num_recursive_calls (int, optional): How many recursive calls have we made. Defaults to 0.
        desired_sizes (List[Coordinate], optional): A list of tile sizes to use. Defaults to [(2, 2), (3, 3)].
        force_random_tilesizes (bool, optional): If True, randomises the tile size selection instead of going in order. Defaults to False.

    Returns:
        World: The generated world
    """
    global failed_steps
    curr = state
    if h_index >= min(number_of_hierarchy_levels, len(list_of_patterns)): return state # we are done

    # Get the appropriate values for this hierarchical level.
    patterns, max_counts, should_randomise_coords, propagate_patterns = (
        list_of_patterns[h_index],
        list_of_max_counts[h_index],
        list_of_should_randomise_coords[h_index],
        list_of_propagate_patterns[h_index],
    )
    is_lowest_level_hierarchy = h_index == number_of_hierarchy_levels - 1

    if num_recursive_calls % 1000 == 0: gc.collect()
    
    if failed_steps >= MAX_STEPS: return curr # we cannot find a solution within the time limit.

    # The tile sizes to iterate over
    tile_sizes = get_tile_sizes_to_use(patterns, is_lowest_level_hierarchy, force_random_tilesizes)
    
    if should_return_given_max_counts(max_counts, num_recursive_calls - recursive_calls_at_start_of_level, state):
        # Here we go to the next hierarchical level, cleaning up first.
        curr = list_of_hierarchy_cleanups[h_index](h_index, curr)
        print(f"Going to hierarchy level {h_index + 1}")
        return recursive_solve_hierarchy(
            curr,
            list_of_patterns=list_of_patterns,
            list_of_max_counts=list_of_max_counts,
            list_of_should_randomise_coords=list_of_should_randomise_coords,
            num_recursive_calls=num_recursive_calls + 1,
            h_index=h_index + 1,
            strict=strict,
            list_of_propagate_patterns=list_of_propagate_patterns,
            save_loc=save_loc,
            number_of_hierarchy_levels=number_of_hierarchy_levels,
            check_cropped_done=check_cropped_done,
            list_of_hierarchy_cleanups=list_of_hierarchy_cleanups,
            desired_sizes=desired_sizes,
            force_random_tilesizes=force_random_tilesizes,
            recursive_calls_at_start_of_level = num_recursive_calls + 1
        )

    if curr.is_done(patterns, check_cropped_done=check_cropped_done): return curr # Here we return because the level is fully generated

    # Now we aim to collapse. We loop over:
    #   - All possible tile sizes
    #   - All possible coordinates (order of entropy/randomly)
    #   - All compatible patterns
    for size_to_use in tile_sizes:
        valid_patterns = [p for p in patterns if p.shape == size_to_use] # the patterns of the appropriate size
        coords = get_coords_to_use(curr, size_to_use, should_randomise_coords, coord_filter)

        for coordinate in coords:
            # Now get patterns that fit here
            curr_slice = get_slice_of_map(curr, coordinate, size_to_use).flatten()

            # Find all of the compatible patterns. This is a List of 3 elements, patterns_flat, probabilities_of_patterns, old_patterns
            compatible_patterns = get_compatible_patterns(valid_patterns, curr_slice)

            if len(compatible_patterns) == 0: # No patterns fit here
                if should_randomise_coords or num_recursive_calls == 0 or (strict and h_index == 0): # Continue, try again
                    continue

                # See if it is possible for any tile size pattern to fit here -- If so, continue, else break.
                if strict and should_continue_when_no_compatible_patterns(curr, coordinate, propagate_patterns, desired_sizes): 
                    continue # Try another size
                else:
                    break # We cannot place anything here, so fail


            compatible_patterns = get_updated_compatible_patterns(curr, compatible_patterns)
            if compatible_patterns == False: continue

            # Now loop over possible patterns until we find a valid one
            all_indices, all_probs, patterns_flat = get_patterns_proper_format(compatible_patterns)
            for _ in range(len(compatible_patterns[0])):
                wrong_old = failed_steps
                def _get_args():
                    return patterns_flat, pat_idx_to_use, save_loc, h_index, num_recursive_calls, coordinate, temp, wrong_old
                failed_steps += 1
                # Sample a pattern
                selected_flat_pattern, pat_idx_to_use = sample_pattern(all_probs, all_indices, patterns_flat)

                # Make a change
                temp = curr.make_new(selected_flat_pattern, coordinate, size_to_use)

                if temp.dict_mapping is not None: temp.dict_mapping_wip[patterns_flat[pat_idx_to_use].global_id] += 1

                _do_log("after_col", *_get_args())

                if temp.is_done(patterns, check_cropped_done=check_cropped_done): return temp  # we are done

                if temp.propagate_constraints(patterns + propagate_patterns, size_to_use, coordinate, be_strict=True) == False:
                    _do_log("propagate_failed", *_get_args())
                    # Failed to propagate, so try something else
                    continue
                
                _do_log("after_propagate", *_get_args())
                
                wrong_old = failed_steps
                # Otherwise, we succeeded to propagate, so recurse
                temp2 = recursive_solve_hierarchy(
                    temp,
                    list_of_patterns=list_of_patterns,
                    list_of_max_counts=list_of_max_counts,
                    list_of_should_randomise_coords=list_of_should_randomise_coords,
                    num_recursive_calls=num_recursive_calls + 1,
                    h_index=h_index,
                    strict=strict,
                    list_of_propagate_patterns=list_of_propagate_patterns,
                    save_loc=save_loc,
                    number_of_hierarchy_levels=number_of_hierarchy_levels,
                    check_cropped_done=check_cropped_done,
                    list_of_hierarchy_cleanups=list_of_hierarchy_cleanups,
                    desired_sizes=desired_sizes,
                    force_random_tilesizes=force_random_tilesizes,
                    recursive_calls_at_start_of_level=recursive_calls_at_start_of_level
                )
                if temp2 is not None:
                    return temp2 # we succeeded, return
                # Failed recurse, try again
                _do_log("after_recurse_failed", *_get_args())
                continue
            if is_lowest_level_hierarchy:
                return None # We failed here
    if curr.is_done(patterns, check_cropped_done=check_cropped_done): return curr
    return None

################################# Logs/Writing Images #################################

def myplot_world_or_pattern(*args, **kwargs):
    # Plots a world or a pattern. First argument is the object and the second argument is the filename to save as.
    obj_to_plot = args[0]
    name = (args[1] + ".txt").replace("/plots/", "/texts/")
    if failed_steps >= 10_000: # Do not plot too many
        return
    if isinstance(obj_to_plot, World):
        helpers.save_world_to_txt_file(obj_to_plot, name)
    else:
        helpers.save_pattern_to_txt_file(obj_to_plot, name)
    if conf.PLOT_IMAGES or failed_steps % 1000 == 0:
        return plot_world_or_pattern(*args, **kwargs)


def plots_on_propagate_failed(save_loc: str, temp: World, h_index: int, counts: int, my_strict: bool):
    # Plots after propagation failed
    myplot_world_or_pattern(
        temp,
        f"{save_loc}/{h_index}/{str(failed_steps).zfill(4)}_{counts}_after_prop_failed",
        rect=temp.bad_coord[::-1] if my_strict else None,
    )
    if DEBUG and my_strict:
        with open(
            f"{save_loc.replace('/plots/', '/texts/')}/{h_index}/{str(failed_steps).zfill(4)}_{counts}_after_prop_failed.json",
            "w+",
        ) as f:
            bc = temp.bad_coord
            json.dump(
                {
                    "superpositions": {
                        str(bc): str(temp.map[bc[0], bc[1]].possible_tiles),
                        str((bc[0] + 1, bc[1])): str(temp.map[bc[0] + 1, bc[1]].possible_tiles),
                        str((bc[0], bc[1] + 1)): str(temp.map[bc[0], bc[1] + 1].possible_tiles),
                        str((bc[0] + 1, bc[1] + 1)): str(temp.map[bc[0] + 1, bc[1] + 1].possible_tiles),
                    }
                },
                f,
            )


def write_pattern_id(save_loc: str, h_index: int, counts: int, patterns_flat: List[InputPattern], pat_idx_to_use: int):
    # Writes the selected pattern id
    DDD = f"{save_loc}/{h_index}/{str(failed_steps).zfill(4)}_{counts}_selectedpat_id.txt".replace("/plots/", "/texts/")
    os.makedirs(os.path.join(*DDD.split("/")[:-1]), exist_ok=True)
    with open(DDD, "w+") as f:
        f.write(str(patterns_flat[pat_idx_to_use].id))


def _do_log(where,
            patterns_flat, pat_idx_to_use, save_loc, h_index, num_recursive_calls, coordinate,
            temp, wrong_old
            ):
    if not DO_LOG: return
    # Logs and writes output files
    if where == "after_col":
        myplot_world_or_pattern(
            patterns_flat[pat_idx_to_use],
            f"{save_loc}/{h_index}/{str(failed_steps).zfill(4)}_{num_recursive_calls}_selectedpat_{coordinate}",
            title=f"Coord = {coordinate}",
        )
        write_pattern_id(save_loc, h_index, num_recursive_calls, patterns_flat, pat_idx_to_use)
        myplot_world_or_pattern(
            temp, f"{save_loc}/{h_index}/{str(failed_steps).zfill(4)}_{num_recursive_calls}_after_coll"
        )
    elif where == "propagate_failed":
        plots_on_propagate_failed(save_loc, temp, h_index, num_recursive_calls, True)
    elif where == "after_propagate":
        myplot_world_or_pattern(
                    temp, f"{save_loc}/{h_index}/{str(failed_steps).zfill(4)}_{num_recursive_calls}_after_prop"
                )
    elif where == "after_recurse_failed":
        myplot_world_or_pattern(
                        temp,
                        f"{save_loc}/{h_index}/{str(wrong_old).zfill(4)}_{num_recursive_calls}_after_recurse_failed",
                    )
    else:
        assert False


################################# Helpers #################################

def should_return_given_max_counts(max_counts: Union[int, float], counts: int, state: World) -> bool:
    """This checks if we should go to the next hierarchical level given the current world's filled to empty ratio.

    Args:
        max_counts (Union[int, float]):
        counts (int): The number of times the recursive function has been called
        state (World):

    Returns:
        bool:
    """
    T = state.num_collapsed_tiles(do_not_count_superpos=True)
    P_BAR.update(np.round(T, 3) - P_BAR.last_print_n)
    
    if max_counts is None:
        return False
    if type(max_counts) == int and counts >= max_counts:
        print(f"Returning because we placed {counts} elements and {max_counts=}")
        return True
    if type(max_counts) == float and T >= max_counts:
        print("Returning because fraction is more than needed")
        return True
    return False

def sample_pattern(all_probs: List[float], all_indices: List[int], patterns_flat: List[InputPattern]) -> Tuple[np.ndarray, int]:
    """Given a list of (unnormalised) probabilities, indices and patterns, return the tiles and the index of the pattern selected.

    Args:
        all_probs (List[float]): 
        all_indices (List[int]): 
        patterns_flat (List[InputPattern]): 

    Returns:
        Tuple[np.ndarray, int]: 
    """
    p = np.array(all_probs)
    p = p / p.sum()
    idx = np.random.choice(len(all_indices), p=p)
    pat_idx_to_use = all_indices[idx]
    all_indices.pop(idx)
    all_probs.pop(idx)
    T = patterns_flat[pat_idx_to_use].tiles.flatten()
    return T, pat_idx_to_use

def get_compatible_patterns(valid_patterns: List[InputPattern], curr_slice: List[SuperPosition]) -> Tuple[List[np.ndarray], List[int], List[InputPattern]]:
    """Given a list of patterns, return [flats], [occurrence], [patterns] that are compatible with the current slice.

    Args:
        valid_patterns (List[InputPattern]): 
        curr_slice (List[SuperPosition]): 

    Returns:
        Tuple[List[np.ndarray], List[int], List[InputPattern]]: 
    """
    return list(
                zip(
                    *[
                        (f, p.occurrence, p)
                        for p in valid_patterns
                        if helpers.is_superposition_compatible_with_pattern(curr_slice, f := p.tiles.flatten())
                    ]
                )
            )

def should_continue_when_no_compatible_patterns(curr: World, coordinate: Coordinate, propagate_patterns: List[InputPattern], desired_sizes: List[Coordinate]) -> bool:
    """Returns True if we should still continue to the next step, and False if we should abort and break out of the current for loop

    Args:
        curr (World): 
        coordinate (Coordinate): 
        propagate_patterns (List[InputPattern]): 
        desired_sizes (List[Coordinate]): 

    Returns:
        bool: 
    """
    for test_size in desired_sizes:
        valid_patterns_prop = [p for p in propagate_patterns if p.shape == test_size]
        valid_patterns_prop = [p for p in valid_patterns_prop if SUPER_POS not in p.tiles.flatten()]
        
        curr_slice_version2 = curr.map[
            tuple(slice(start, start + length, None) for start, length in zip(coordinate, test_size))
        ]
        if curr_slice_version2.shape != test_size: continue  # is at the edge
        curr_slice_version2 = curr_slice_version2.flatten()
        
        if all([c.is_collapsed for c in curr_slice_version2]): # This is collapsed, so we are good
            return True

        compatible_patterns_for_propagates = get_compatible_patterns(valid_patterns_prop, curr_slice_version2)
        if len(compatible_patterns_for_propagates) > 0: 
            return True
    return False

def get_updated_compatible_patterns(curr: World, compatible_patterns: Tuple[List[np.ndarray], List[int], List[InputPattern]]) -> List[InputPattern]:
    """Give the patterns, if we have placed fewer than desired of a specific subset, we return only those.

    Args:
        curr (World): 
        compatible_patterns (Tuple[List[np.ndarray], List[int], List[InputPattern]]): 

    Returns:
        List[InputPattern]: 
    """
    if curr.dict_mapping is not None:
        # Only make compatible those patterns we have placed few of.
        compatible_patterns = list(zip(*compatible_patterns))
        proper_compatible = []
        for tup in compatible_patterns:
            curr_count = curr.dict_mapping_wip[tup[2].global_id]
            max_count = curr.dict_mapping.get(tup[2].global_id, float("inf"))
            if curr_count < max_count: proper_compatible.append(tup)
        compatible_patterns = proper_compatible
        if len(compatible_patterns) == 0: return False
        compatible_patterns = list(zip(*compatible_patterns))
        assert len(compatible_patterns) == 3
    return compatible_patterns

def get_patterns_proper_format(compatible_patterns: Tuple[List[np.ndarray], List[int], List[InputPattern]]) -> Tuple[List[int], List[np.ndarray], List[InputPattern]]:
    # Just reformats
    all_indices     = list(range(len(compatible_patterns[0])))
    all_probs       = [i for i in compatible_patterns[1]]
    patterns_flat   = compatible_patterns[2]
    return all_indices, all_probs, patterns_flat

def get_coords_to_use(curr: World, size_to_use: Coordinate, should_randomise_coords: bool, coord_filter: FilterCoord) -> List[Coordinate]:
    """This returns a list of coordinates we should loop over, in order. It possibly shuffles the list if `should_randomise_coords` is True

    Args:
        curr (World): 
        size_to_use (Coordinate): 
        should_randomise_coords (bool): 
        coord_filter (FilterCoord): 

    Returns:
        List[Coordinate]: 
    """
    coords = curr.get_coords_to_iterate_over(size_to_use, coord_filter)
    if should_randomise_coords and np.random.rand() < RANDOM_PERCENT: random.shuffle(coords) # possibly randomise the coordinates
    return coords

def get_tile_sizes_to_use(patterns: List[InputPattern], is_lowest_level_hierarchy: bool, force_random_tilesizes: bool) -> List[Coordinate]:
    """Returns a list of tile sizes that we can use, in order -- either random, or in order of decreasing size if `is_lowest_level_hierarchy` is True.

    Args:
        patterns (List[InputPattern]): 
        is_lowest_level_hierarchy (bool): 
        force_random_tilesizes (bool): 

    Returns:
        List[Coordinate]: 
    """
    tile_sizes = list(set([p.shape for p in patterns]))
    np.random.shuffle(tile_sizes)
    if is_lowest_level_hierarchy and not force_random_tilesizes:
        tile_sizes = sorted(tile_sizes)[::-1]
    return tile_sizes

def get_slice_of_map(curr: World, loc: Coordinate, size: Coordinate) -> np.ndarray: # of SuperPositions
    return curr.map[
        tuple(slice(start, start + length, None) for start, length in zip(loc, size))
    ]
