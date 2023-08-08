from typing import List, Tuple

from hwfc.utils.classes import InputPattern


def get_all_subpatterns(pattern: InputPattern, size: Tuple[int, int]) -> List[InputPattern]:
    """Get all subpatterns of a given size from a pattern.

    Args:
        pattern (InputPattern): 
        size (Tuple[int, int]): 

    Returns:
        List[InputPattern]: 
    """    """"""
    t = set()
    for i in range(pattern.shape[0] - size[0] + 1):
        for j in range(pattern.shape[0] - size[1] + 1):
            slice = InputPattern(pattern.tiles[i:i + size[0], j:j + size[1]])
            t.add(slice)
    return list(t)