
from typing import List, Tuple

import numpy as np


class Tile:
    # A tile class that can hold a value, i.e., a tile type.
    value: int
    def __init__(self, value: int, num_total_tiles: int) -> None:
        self.value = value
        self.num_total_tiles = num_total_tiles
        assert value < num_total_tiles, f"Make `num_total_tiles` (currently {num_total_tiles}) larger than {value}"
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def __eq__(self, __o: object) -> bool:
        return self.value == __o.value
    
    def __repr__(self) -> str:
        return f"T({self.value})"
    
    def __mul__(self, other): # Just a hack to make it work with numpy.kron
        return self


class InputPattern:
    # This is an input pattern, i.e., a nxk pattern of collapsed tiles. 
    # This is used to represent both the designed examples as well as the overlapping patterns that are extracted from these examples.
    occurrence: int # how many times this pattern occurs.
    tiles: np.ndarray #[Tile]
    shape: Tuple[int, int]
    def __init__(self, tiles: List[List[Tile]], occurrence: int = 1) -> None:
        self.id = -1
        self.tiles = np.array(tiles)
        self.shape = tiles.shape
        # One hot encoding
        tot = tiles.flatten()[0].num_total_tiles
        def _one_hot(n, tot):
            a = np.zeros(tot, dtype=np.bool8)
            a[n] = True
            return a
        self.flat_vals = np.array([_one_hot(t.value, tot) for t in (self.tiles).flatten()])
        self.occurrence = occurrence
        self.global_id = None
        self.index_array = np.array([t.value for t in self.tiles.flatten()])
    
    def __hash__(self) -> int:
        return hash(tuple(x for x in self.tiles.flatten()))

    def __eq__(self, __o: object) -> bool:
        return (self.tiles).flatten().tolist() == (__o.tiles).flatten().tolist()
    
    def set_global_id(self, id):
        self.global_id = id