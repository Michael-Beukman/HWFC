from typing import Tuple, Union
import numpy as np

from hwfc.utils.classes import Tile
Example = np.ndarray
# This needs to be larger than the number of tiles, otherwise bad things happen
TOT_TILES = 50

# Several distinct super positions to deal with our hierarchical setup.
SUPER_POS = Tile(TOT_TILES-1, TOT_TILES)
ALL_SUPER_POSITIONS = [Tile(TOT_TILES-i, TOT_TILES) for i in range(2, 10)]

# a 2D or 3D coordinates
Coordinate = Union[Tuple[int, int], Tuple[int, int, int]]

FlatPattern = np.ndarray
