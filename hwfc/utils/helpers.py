from collections import defaultdict
import copy
import functools
import itertools
import os
import sys
from typing import List, Set, Tuple
import numpy as np
from hwfc.utils.classes import InputPattern, Tile
from hwfc.utils.conf import MAX_PATTERN_OCCURRENCE
from hwfc.utils.mytypes import ALL_SUPER_POSITIONS, SUPER_POS, TOT_TILES
import hwfc.world as world

def read_3d_example(filename: str) -> InputPattern:
    """Reads a 3D example from a filename

    Args:
        filename (str): 

    Returns:
        InputPattern: 
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        shape = tuple(map(int, lines[1].split()))
        splits = [l for l in lines[2].split(",") if l]
        lines = np.array(list(map(lambda x: Tile(int(x), TOT_TILES), splits)))
        return InputPattern(lines.reshape(shape))

CURR_PAT_ID_VAL = 0
pattern_ids = {}

def load_example_from_file(filename: str) -> InputPattern:
    """This loads an example from a file, returning an input pattern.

    Args:
        filename (str): The file containing the example.

    Returns:
        InputPattern: The example.
    """
    global CURR_PAT_ID_VAL
    ss = filename
    
    # Get a new unique ID, either the same if a rotated version of this pattern has already been read, or a new, monotonically increasing ID.
    for i in range(10): ss = ss.replace(f"Rot{i}", "Rot")
    if ss in pattern_ids:
        curr = pattern_ids[ss]
    else:
        pattern_ids[ss] = CURR_PAT_ID_VAL
        curr = CURR_PAT_ID_VAL
        CURR_PAT_ID_VAL += 1
    
    def inner():
        arr = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            if lines[0].strip() == '3d': return read_3d_example(filename)
            
            for row in lines:
                row = row.strip().replace("[", '').replace("]", '')
                if not row: continue
                tmp = []
                for col in row.split(","):
                    if col == '': continue
                    col = int(col)
                    tmp.append(SUPER_POS if col == -1 else Tile(col, TOT_TILES))
                arr.append(tmp)
        return InputPattern(np.array(arr))
    ans = inner()
    ans.set_global_id(curr)
    return ans

def rotate_pattern(pattern: InputPattern) -> List[InputPattern]:
    """Returns a list of patterns containing all of the 4 90 degree rotations of this pattern.

    Args:
        pattern (InputPattern): 

    Returns:
        List[InputPattern]: 
    """
    patterns = [pattern] + [InputPattern((np.rot90(pattern.tiles, k)).tolist()) for k in range(1, 4)]
    return patterns

def make_nd_range_to_iterate_over(vals: List[int]) -> List[tuple]:
    """Given a list of integers (say [2, 3, 4]), returns all combination of (i, j, k) such that 0 <= i < 2, 0 <= j < 3, 0 <= k < 4, etc.
        This is useful, say, if we have a radius of 4 units in 2 dimensions and want to iterate through all tiles within that radius.

    Args:
        vals (List[int]): 

    Returns:
        List[tuple]:
    """
    ranges = [list(range(i)) for i in vals]
    return itertools.product(*ranges)

def get_patterns_from_tilemap(tilemap: InputPattern, tile_size=(3, 3), do_rotate: bool = False, ignore_superpos=False, max_occurrence=MAX_PATTERN_OCCURRENCE) -> List[InputPattern]:
    """This takes a tilemap and returns a list of patterns of size `tile_size` that are extracted from this tilemap.

    Args:
        tilemap (InputPattern): The example to extract patterns from.
        tile_size (tuple, optional): The size of patterns to extract. We use overlapping windows. Defaults to (3, 3).
        do_rotate (bool, optional): If true, uses all of the 4 rotations of the pattern, otherwise just this one. Defaults to False.
        ignore_superpos (bool, optional): If this is true, we do not return any patterns where a superposition is in. Defaults to False.
        max_occurrence (int, optional): The maximum number of times a pattern can occur. Defaults to MAX_PATTERN_OCCURRENCE.

    Returns:
        List[InputPattern]: A list of all patterns of size `tile_size`.
    """
    assert len(tilemap.tiles.shape) == len(tile_size) # confirms that we do not try to e.g. extract 2D patterns from a 3D example.
    
    
    B = make_nd_range_to_iterate_over([i - size + 1 for i, size in zip(tilemap.shape, tile_size)])
    patterns = []
    count_of_patterns = defaultdict(lambda: 0)
    for coord in B: # iterate over all coordinates.
        sl = tuple([slice(c, c + s, None) for c, s in zip(coord, tile_size)])
        _arr = tilemap.tiles[sl]
        
        if  ignore_superpos and SUPER_POS in  _arr.flatten(): # ignore patterns with super pos tiles in them.
            continue
        
        if do_rotate:
            patterns.extend(rotate_pattern(InputPattern(_arr)))
        else:
            patterns.extend([(InputPattern(_arr))])
    
    total = 0
    for p in patterns: 
        count_of_patterns[p] += 1
        total += 1
    for p, count in count_of_patterns.items(): p.occurrence = min(count, max_occurrence) # cap the number of occurrences at a specific value.
    return list(set(patterns)) # only unique patterns

def get_initial_map_from_example(tilemap: InputPattern, all_tile_options: List[Tile]) -> "world.World":
    """Given an example, return an already-collapsed world. The tiles where the example is labelled as -1 (SUPER_POS) will be left uncollapsed.

    Args:
        tilemap (InputPattern): 
        all_tile_options (List[Tile]): All of the options that tiles can take.

    Returns:
        world.World: 
    """
    arr = tilemap.tiles
    w, h = len(arr), len(arr[0])
    map = world.World(all_tile_options, (h, w))
    for i, row in enumerate(map.map):
        for j, col in enumerate(row):
            if arr[i][j].value < 0: continue
            map.map[i][j].collapse(arr[i][j])
    return map

def get_all_tile_options_from_pattern(pattern: InputPattern) -> Set[Tile]:
    """From a pattern, return all possible tiles that are contained in this pattern.

    Args:
        pattern (InputPattern):

    Returns:
        Set[Tile]:
    """
    ans = set()
    for t in pattern.tiles.flatten():
        ans.add(t)
    return ans

def get_all_tile_options_from_all_patterns(patterns: np.ndarray) -> Set[Tile]:
    """From an array of patterns, return all possible tiles that are contained in these patterns.

    Args:
        patterns (np.ndarray): 

    Returns:
        Set[Tile]: 
    """
    ans = set()
    flats: List[InputPattern] = patterns.flatten()
    for p in flats:
        ans |= get_all_tile_options_from_pattern(p)
    ans.add(SUPER_POS)
    ans |= set(ALL_SUPER_POSITIONS)
    return ans

def is_superposition_compatible_with_pattern(list_of_superpos: List[world.SuperPosition], list_of_tiles: List[Tile]) -> bool:
    """
    Returns true if this superposition slice is compatible with the pattern represented by the list of tiles. Note, this assumes both are flattened.
    Compatible means that one possible collapse of the superposition tiles is the same as the pattern.
    Args:
        list_of_superpos (np.ndarray): List[Superposition]
        list_of_tiles (np.ndarray): List[Tile]

    Returns:
        bool: True if these superpositions is compatible with the list of tiles
    """
    assert len(list_of_superpos) == len(list_of_tiles), f"Sizes are not the same {len(list_of_superpos)} != {len(list_of_tiles)}"
    for a, b in zip(list_of_superpos, list_of_tiles):
        if b in [SUPER_POS] + ALL_SUPER_POSITIONS and a.possible_tiles == [SUPER_POS]:
            return False
        if b not in a.possible_tiles: 
            if b not in [SUPER_POS] + ALL_SUPER_POSITIONS:
                return False
    return True


def optimised_is_superposition_compatible_with_pattern_indices(flat_slice: List[world.SuperPosition], all_patterns_vals: List[np.ndarray], patterns: List[Tuple[int, InputPattern]], coord_list=False) -> List[InputPattern]:
    """This, given a list of superpositions and a bunch of patterns, return the patterns that are compatible with this superposition list in a relatively quick way.

    Args:
        flat_slice (List[world.SuperPosition]): The superpositions
        all_patterns_vals (List[np.ndarray]): This is a list of pattern representations, indicating which tiles they have at each location.
        patterns (List[InputPattern]): This is the list of patterns, with the same length as `all_patterns_vals`

    Returns:
        List[InputPattern]: The compatible patterns
    """
    
    # Shape = (chunk_size, possible_tiles)
    flat_one_hot = np.array([p.flat_vals.copy() for p in flat_slice])
    
    flat_one_hot[flat_one_hot[:, SUPER_POS.value] == 1, :] = 1

    new_one = np.take_along_axis(flat_one_hot, all_patterns_vals.T, axis=1).T
    c3 = new_one.all(axis=-1)
    if coord_list:
        return np.arange(len(patterns))[c3]
    a1 = [patterns[i] for i in range(len(c3)) if c3[i]]
    return a1



dic_memo = {}
@functools.cache
def _cached_unravel_index(i: int, shape):
    return np.unravel_index(i, shape)

@functools.cache
def _cached_ravel_multi_index(i: tuple, shape):
    return np.ravel_multi_index(i, shape)

def get_superpositions_from_multiple_tiles(patterns: List[List[Tile]], pattern_shapes) -> List[world.SuperPosition]:
    """Given a list of patterns, create a superposition from all of them. E.g. if pattern 1 has a 0 in the top left and pattern 2 has a 1, then the tile in the top left will be a superposition of {0, 1}

    Args:
        patterns (List[List[Tile]]): The list of patterns to use

    Returns:
        world.SuperPosition:
    """

    ids = tuple(p[0] for p in patterns)
    if ids in dic_memo: 
        return (dic_memo[ids])
    patterns = [p[1] for p in patterns]
    lens = [len(p) for p in patterns]
    superpos = []
    min_len_idxs = np.argmin(lens)
    len_to_iterate_over = lens[min_len_idxs]
    superpos_shape_to_use = pattern_shapes[min_len_idxs]

    for i in range(len_to_iterate_over):
        t = set()
        for pat, pat_shape in zip(patterns, pattern_shapes):
            # We now have a pattern which is a flat array and a pat_shape
            # We also have an index 
            superpos_idx   = _cached_unravel_index    (i, superpos_shape_to_use)
            idx_in_pattern = _cached_ravel_multi_index(superpos_idx, pat_shape)

            if idx_in_pattern >= len(pat) or pat[idx_in_pattern] in t: continue
            t.add(pat[idx_in_pattern])
        superpos.append(world.SuperPosition(list(t)))
    dic_memo[ids] = superpos
    return superpos


def random_from_list(li, return_idx=False):
    """Returns a random item from a list, optionally the index as well

    Args:
        li (_type_): 
        return_idx (bool, optional):. Defaults to False.

    """
    idx = np.random.choice(np.arange(len(li)))
    if return_idx:
        return li[idx], idx
    return li[idx]

def check_dir_exists(filename: str):
    """Makes the parent directory if it does not already exist.

    Args:
        filename (str): 
    """
    dirs = os.path.join(*filename.split(os.sep)[:-1])
    os.makedirs(dirs, exist_ok=True)

def save_world_to_txt_file(world: world.World, filename: str):
    """This saves a world to a text file

    Args:
        world (world.World): 
        filename (str): 
    """
    if len(world.map.shape) == 3: 
        world = copy.deepcopy(world)
        world.map = world.map.reshape((world.map.shape[0], world.map.shape[1] * world.map.shape[2]))
    s = ""
    for j, row in enumerate(world.map):
        tmp = ''
        for i, col in enumerate(row):
            if not col.is_collapsed:
                T = str(-1)
            else:
                T = str(col.possible_tiles[0].value)
            tmp += T + ","
        s += tmp + ("\n" if (j != len(world.map) -1) else '')
    
    check_dir_exists(filename)
    with open(filename, 'w+') as f:
        f.write(s)
        
def save_pattern_to_txt_file(pat: world.InputPattern, filename: str):
    if len(pat.tiles.shape) == 3: 
        pat = copy.deepcopy(pat)
        pat.tiles = pat.tiles.reshape((pat.tiles.shape[0], pat.tiles.shape[1] * pat.tiles.shape[2]))
    s = ""
    for j, row in enumerate(pat.tiles):
        tmp = ''
        for i, col in enumerate(row):
            T = str(col.value)
            tmp += T + ","
        s += tmp + ("\n" if (j != len(pat.tiles) -1) else '')
    
    check_dir_exists(filename)
    with open(filename, 'w+') as f:
        f.write(s)

def save_3d_example_to_file(pat: InputPattern, filename: str):
    shape = pat.tiles.shape
    s = "3d\n" + " ".join(map(str, shape)) + "\n"
    for j, tile in enumerate(pat.tiles.flatten()):
        s += str(tile.value) + ","
    
    check_dir_exists(filename)
    with open(filename, 'w+') as f:
        f.write(s)
