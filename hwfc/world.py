from collections import defaultdict
import copy
import itertools
from typing import Callable, DefaultDict, Dict, List, Set, Tuple, Union

import numpy as np

from hwfc.utils.classes import InputPattern, Tile
import hwfc.utils.helpers as helpers
from hwfc.utils.mytypes import ALL_SUPER_POSITIONS, SUPER_POS, TOT_TILES, Coordinate

FilterCoord = Callable[[Coordinate], bool]
def default_filter_coord(c): return True


def get_coord_filter_specific_height(height: int):
    def filt(coord: Coordinate):
        return coord[1] == height
    return filt


class SuperPosition:
    """A superposition is a list of possible tiles
    """

    def __init__(self, all_tile_options: List[Tile]) -> None:
        """Given a list of tiles that this superposition can be in, constructs it as being possible to be each of them

        Args:
            all_tile_options (List[Tile]): 
        """
        self.possible_tiles = sorted(all_tile_options, key=lambda x: x.value)
        self.is_collapsed = len(all_tile_options) == 1
        self.flat_vals = self._get_flat()

    def collapse(self, tile_to_choose: Tile = None):
        """Collapses into a random possible tile, or to the specific tile given

        Args:
            tile_to_choose (Tile, optional): . Defaults to None.
        """
        self.is_collapsed = True
        if tile_to_choose is not None:
            assert tile_to_choose in self.possible_tiles or tile_to_choose in ALL_SUPER_POSITIONS + [SUPER_POS]
            self.possible_tiles = [tile_to_choose]
        else:
            self.possible_tiles = [np.random.choice(self.possible_tiles)]
        self.flat_vals = self._get_flat()

    def _get_flat(self):
        # An optimisation that represents this as a boolean array of size |all_tile_options|
        tot = self.possible_tiles[0].num_total_tiles
        a = np.zeros(tot, dtype=np.bool8)
        for n in self.possible_tiles:
            a[n.value] = True
        return a

    def set(self, possible_tiles: List[Tile], is_collapsed: bool):
        """Sets this object's possible tiles and collapsed variables. Collapsed means that there is only one possible option.

        Args:
            possible_tiles (List[Tile]): 
            is_collapsed (bool): 
        """
        self.possible_tiles = possible_tiles
        self.is_collapsed = is_collapsed
        self.flat_vals = self._get_flat()


def mylist(): 
    return []
def myzero():
    return 0

class World:
    """
        Represents the map within which the superpositions can evolve.
    """
    map: np.ndarray  # [SuperPosition]

    def __init__(self, list_of_tiles: List[Tile], shape: List[int] = (10, 10)):
        """Creates this world object, given a set of tiles that are allowed to be in the map and the map's size.

        Args:
            list_of_tiles (List[Tile]): 
            shape (List[int], optional): . Defaults to (10, 10).
        """
        self.shape = shape
        list_of_tiles = list(list_of_tiles)
        self.map = []
        self.map = np.zeros(shape, dtype=SuperPosition)

        for coord in helpers.make_nd_range_to_iterate_over(shape):
            self.map[coord] = SuperPosition(copy.deepcopy(list_of_tiles))

        self._is_done = False
        self.all_tile_options: Set[Tile] = set(list_of_tiles)

        self.compatible_patterns_cache: Dict[Tuple[Coordinate,
                                                   Coordinate], List[InputPattern]] = {}

        self.good_coord = None
        self.used_patterns: DefaultDict[Tuple[Coordinate,
                                       Coordinate], List[InputPattern]] = defaultdict(mylist)
        self.dict_mapping = None
        self.dict_mapping_wip = defaultdict(myzero)

    def num_collapsed_tiles(self, fraction=True, do_not_count_superpos=False) -> Union[int, float]:
        """This returns how filled up the level is with collapsed tiles.

        Args:
            fraction (bool, optional): If true, returns a fraction of the total tiles. Defaults to True.
            do_not_count_superpos (bool, optional): If True, does not count superposition tiles. Defaults to False.

        Returns:
            Union[int, float]: Either an integer number of tiles filled or fraction of the map
            
        """
        ans = 0
        good_size = 0
        for t in self.map.flatten():
            good_size += 1
            if t.is_collapsed: 
                if do_not_count_superpos and t.possible_tiles[0] in ALL_SUPER_POSITIONS + [SUPER_POS]:
                    good_size -= 1
                    continue
                ans += 1
        if fraction: 
            ans /= self.map.size
        return ans
    
    def _get_lowest_entropy(self, tile_size: Coordinate,
                            coord_filter: FilterCoord = default_filter_coord) -> Dict[int, List[Coordinate]]:
        """This returns an entropy dictionary, basically saying how much entropy specific coordinates have.

        Args:
            tile_size (Coordinate): The size of the pattern to look at, e.g. 2x2 will be a 2x2 pattern, etc.
            coord_filter (FilterCoord): Which coordinates should be looked at, by default all of them

        Returns:
            Dict[int, List[Coordinate]]: {entropy: List of tile coords with this entropy}
        """
        
        # iterate over overlapping windows of the given size.
        number_to_iterate_over = np.array(self.shape) - np.array(tile_size) + 1
        NUM_EXPECTED = np.prod(tile_size)
        def _calc_entropy_of_slice(start_pos: Coordinate):
            entropy = 0
            slice_coords = tuple([slice(a, a + length, None)
                                    for a, length in zip(start_pos, tile_size)])
            maps = self.map[slice_coords].flatten()
            if len(maps) < NUM_EXPECTED: 
                return float('inf') # a boundary tile so ignore.
            
            # A way to get entropy, roughly how many options there are in total for this pattern.
            entropy = sum(len(tile.possible_tiles) for tile in maps)
            if entropy == maps.size: # This is already collapsed
                entropy = float('inf')
            return entropy

        all_entropies_to_coords = defaultdict(mylist)

        
        # Iterate over all of the coords in a dimension-agnostic way.
        all_coords_to_iterate_over = helpers.make_nd_range_to_iterate_over(number_to_iterate_over)
        
        for current_tile_coord in all_coords_to_iterate_over:
            # current_tile_coord = (y, x, [z])
            if not coord_filter(current_tile_coord): continue
            ent = _calc_entropy_of_slice(current_tile_coord)
            if ent == float('inf'): continue
            all_entropies_to_coords[ent].append(current_tile_coord)
        return dict(all_entropies_to_coords)

    def get_coords_to_iterate_over(self, size_to_use: Coordinate, coord_filter: FilterCoord) -> List[Coordinate]:
        """This returns a list of coordinates to iterate over, given a size and a filter. In particular, it returns the top-left of each overlapping window of size `size_to_use` that satisfies the filter.

        Args:
            size_to_use (Coordinate): 
            coord_filter (FilterCoord): 

        Returns:
            List[Coordinate]: In sorted order, from lowest to highest entropy.
        """
        dic_of_entropies = self._get_lowest_entropy(size_to_use, coord_filter)
        possible_entropies = sorted(dic_of_entropies.keys())
        return sum([dic_of_entropies[i] for i in possible_entropies], [])
    

    def make_new(self, selected_flat_pattern, coordinate: Coordinate, size_to_use: Coordinate) -> "World":
        """Does not change this object, rather returns a new one where the selected pattern is collapsed at the particular coordinate.

        Args:
            selected_flat_pattern (_type_): 
            coordinate (Coordinate): 
            size_to_use (Coordinate): 

        Returns:
            World: New world with this change.
        """
        new = copy.deepcopy(self)
        curr_slice = new.map[tuple(slice(start, start + length, None) for start, length in zip(coordinate, size_to_use))].flatten()
        for a, b in zip(curr_slice, selected_flat_pattern):
            if a.is_collapsed:
                continue
            a.collapse(b)
        return new

    def propagate_constraints(self, patterns: List[InputPattern], tile_size: Coordinate, collapse_location: Coordinate, do_collapse_empty: bool = True, max_size_to_consider: int = 4, be_strict: bool=False, return_counts=False,
                              queue_override=None) -> Union[bool, Tuple[bool, int]]:
        """Now, after we have collapsed, we can propagate the constraints. This pretty much adjusts the superpositions of the different tiles so that they remain compatible with the changes made in the previous step in a breadth-first way.

        Args:
            patterns (List[InputPattern]): The list of allowed patterns
            tile_size (Coordinate): The size of the tile that we collapsed previously
            collapse_location (Coordinate): The location we collapsed at previously
            do_collapse_empty (bool, optional): If this is true, we collapse the tiles that must be collapsed. Defaults to True.
            max_size_to_consider (int, optional): The maximum tile size to consider when finding tiles that could be affected by change. Defaults to 10.
            return_counts (bool, optional): If true, returns the number of tiles that changed. Defaults to False.
        
        Returns:
            Union[bool, Tuple[bool, int]]: If true, the propagation was successful. If false, it was not.
        """
        
        positions = []
        arr_curr_loc = np.array(collapse_location)
        sizes = [list(range(r)) for r in tile_size]

        for new_loc in itertools.product(*sizes):
            positions.append(tuple(np.array(new_loc) + arr_curr_loc))
        
        # Get the starting tiles to look at.
        if queue_override is not None and len(queue_override) > 0:
            queue_tiles = queue_override
        else:
            queue_tiles = self._get_new_positions_to_care_about(
                positions, max_size=max_size_to_consider)

        tile_sizes = set([p.shape for p in patterns])
        patterns_flat_dic = {size_to_use: [(p.id, p.tiles.flatten()) for p in patterns if p.shape == size_to_use] for size_to_use in tile_sizes}
        
        
        all_flat_values = {size_to_use: np.array([p.flat_vals for p in patterns if p.shape == size_to_use]) for size_to_use in tile_sizes}
        all_pattern_indices = {size_to_use: np.array([p.index_array for p in patterns if p.shape == size_to_use]) for size_to_use in tile_sizes}
        
        all_actual_patterns = {size_to_use: [p for p in patterns if p.shape == size_to_use] for size_to_use in tile_sizes}

        # Go, in a BFS way, through the list of tiles
        visited = set()
        queue_tiles_set = set(queue_tiles)
        self.bad_coord = None
        curr_prop_count = 0
        
        while len(queue_tiles):
            current_pos = queue_tiles.pop(0)
            queue_tiles_set.remove(current_pos)
            visited.add(current_pos)

            invalid_superpositions = 0
            for S in tile_sizes:
                tiles = self._single_propagate_iteration(
                    current_pos, tile_sizes, S, patterns_flat_dic, all_flat_values, all_actual_patterns=all_actual_patterns, do_collapse_empty=do_collapse_empty,
                    all_pattern_indices=all_pattern_indices)
                if tiles is not None and len(tiles) > 0:
                    curr_prop_count += 1
                    new = self._get_new_positions_to_care_about(tiles, max_size=max_size_to_consider)
                    for t in new:
                        if t not in queue_tiles_set and t not in visited:
                            queue_tiles.append(t)
                            queue_tiles_set.add(t)
                elif tiles is None:
                    invalid_superpositions += 1
            
            if invalid_superpositions >= len(tile_sizes): 
                self.bad_coord = current_pos
                return False
                
        if invalid_superpositions < len(tile_sizes):
            self.good_coord = None
            if return_counts:
                return True, curr_prop_count
            return True
        return False

    def _single_propagate_iteration(self, current_pos: Coordinate,
                              tile_sizes: Set[Coordinate], tile_size: Coordinate,
                              patterns_flat_dic: Dict[Coordinate, List[List[Tile]]], all_flat_values: Dict[Coordinate, List[np.ndarray]], 
                              all_actual_patterns: Dict[Coordinate, InputPattern],
                              all_pattern_indices,
                              do_collapse_empty: bool = True) -> Union[None, List[Coordinate]]:
        """This performs a single propagate iteration, given a current coordinate and some tile sizes.

        Args:
            current_pos (Coordinate): The coord that we must investigate.
            tile_sizes (Set[Coordinate]): The set of tile sizes that we consider
            tile_size (Coordinate): The tile size that did collapse.
            patterns_flat_dic (Dict[Coordinate, List[List[Tile]]]): This is a dict (size: list of flat patterns with this size)
            all_flat_values (Dict[Coordinate, List[np.ndarray]]): This is a dict (size: flat vals of patterns) -- for optimised evaluation
            all_actual_patterns (Dict[Coordinate, InputPattern]): This is a dict (size: patterns) 
            do_collapse_empty (bool, optional): If this is true, we collapse empty tiles. Defaults to True.

        Returns:
            Union[None, List[Coordinate]]: Either None if it failed or a list of coordinates that changed.
        """
        compatibles = []
        pattern_shapes = []
        has_all_collapsed = True
        _total = 0
        tile_sizes = sorted(tile_sizes)[::-1] # big first
        
        for size_to_use in tile_sizes:

            current_end_pos = tuple(
                s + l for s, l in zip(current_pos, size_to_use))
            if self.is_out_of_bounds(current_end_pos):
                continue
            patterns_flat = patterns_flat_dic[size_to_use]
            all_patterns_vals = all_flat_values[size_to_use]
            actual_patterns = all_actual_patterns[size_to_use]
            
            proper_indices = all_pattern_indices[size_to_use]

            slice_to_select = tuple(
                [slice(start, end, None) for start, end in zip(current_pos, current_end_pos)])
            flat_slice: List[SuperPosition] = self.map[slice_to_select].flatten()

            has_all_collapsed = has_all_collapsed and all(
                [t.is_collapsed for t in flat_slice])
            if has_all_collapsed:
                continue

            old_compat_flat, old_compat_pat, old_actual_pats = patterns_flat, all_patterns_vals, actual_patterns
            if len(old_compat_flat) == 0:
                continue

            # Here: If the `old_compat_pat` has a length of zero, that means we likely have the issue with unsatisfiable constraints.            
            idxs = helpers.optimised_is_superposition_compatible_with_pattern_indices(flat_slice, proper_indices, old_compat_flat, coord_list=True)

            new_compat_flat = [old_compat_flat[i] for i in idxs]
            new_actual_pats = [old_actual_pats[i] for i in idxs]
            compatibles += new_compat_flat
            pattern_shapes += [p.shape for p in new_actual_pats]
            
            _total += len(patterns_flat)

        if has_all_collapsed:
            return []

        if len(compatibles) == _total:
            # all of the ones are compatible, so it won't reduce the superposition at all.
            return []
        # check if any of these tiles are collapsed

        if len(compatibles) == 0:  # Here we ignore invalid constraints
            return None

        tiles = []
        if not do_collapse_empty:
            return tiles
        # now make superposition from this list
        superpos = helpers.get_superpositions_from_multiple_tiles(compatibles, pattern_shapes)
        for lll, (a, b) in enumerate(zip(flat_slice, superpos)):
            coord = tuple(np.unravel_index(lll, size_to_use))
            if a.is_collapsed:
                continue
            if a.possible_tiles == b.possible_tiles:
                continue
            a.set(b.possible_tiles, b.is_collapsed)
            tiles.append(coord)
        return tiles

    def _get_new_positions_to_care_about(self, current_positions: List[Coordinate], max_size: int) -> List[Coordinate]:
        """Pretty much, given a list of coordinates, and a maximum size, this function returns a list of coordinates that are inside the convex hull of the given coordinates, extended by `max_size` in each direction.

        Args:
            current_positions (List[Coordinate]): Current coordinates
            max_size (int): The largest size to look out.

        Returns:
            List[Coordinate]: New coordinates.  
        """
        # All directions, up down left right, etc. But in an optimised way
        DDD = len(current_positions[0])
        directions = [[-1, 1] for _ in range(len(current_positions[0]))]
        vals_to_iterate_over = [list(range(1, max_size))
                                for _ in range(len(current_positions[0]))]
        all_dirs_np = np.array(list(itertools.product(*directions)))
        all_dirs_vals = np.array(
            list(itertools.product(*vals_to_iterate_over)))
        all_offsets = all_dirs_np[None] * all_dirs_vals[:, None]
        all_offsets = all_offsets.reshape(-1, DDD)
        all_curr_positions = np.array(current_positions)
        all_candidates = all_curr_positions[None] + all_offsets[:, None]
        all_candidates = all_candidates.reshape(-1, DDD)

        aa = (all_candidates > np.array(self.shape)[None]).any(axis=1)
        bb = (all_candidates < np.zeros((1, DDD))).any(axis=1)
        all_candidates = all_candidates[~np.logical_or(
            aa, bb)].reshape(-1, DDD)

        all_candidates = np.unique(all_candidates, axis=0)
        numpy_ans = (([tuple(t) for t in all_candidates]))
        return numpy_ans

    def is_out_of_bounds(self, coord: Coordinate) -> bool:
        return any(nc < 0 or nc >= self.shape[i] + 1 for i, nc in enumerate(coord))

    def is_done(self, patterns: List[InputPattern], check_cropped_done=False) -> bool:
        if check_cropped_done: return self.get_cropped().is_done(patterns)
        pp = patterns[0].shape
        return len(self._get_lowest_entropy(pp)) == 0

    def _put_superpositions_back(self, super_pos_tile: Tile = SUPER_POS) -> List[Coordinate]:
        """Given a super position tile, this function changes this object in place by replacing all {SUPER_POS} with {TILE_0, TILE_1, ...}

        Args:
            super_pos_tile (Tile, optional): . Defaults to SUPER_POS.

        Returns:
            List[Coordinate]: 
        """
        all_changed_things = []
        T = list(self.all_tile_options - {SUPER_POS} - set(ALL_SUPER_POSITIONS))
        for coord in helpers.make_nd_range_to_iterate_over(self.shape):
            col = self.map[coord]
            if col.possible_tiles == [super_pos_tile]:
                col.set(T, False)
                all_changed_things.append(coord)
        return all_changed_things
    
    def get_cropped(self, n: int=1) -> "World":
        """Returns a cropped version of this world

        Args:
            n (int, optional): The number of rows/cols to crop at each edge. Defaults to 1.

        Returns:
            World: The cropped world, does NOT change this one in place
        """
        new = copy.deepcopy(self)
        new.map = new.map[n:-n, n:-n]
        new.shape = tuple((s - 2 * n for s in new.shape))
        return new
