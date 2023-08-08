from natsort import natsorted
from collections import defaultdict
import glob
import pathlib
from typing import Callable
import numpy as np
from PIL import Image
import imagehash

from hwfc.utils.classes import InputPattern
from hwfc.utils.helpers import load_example_from_file
from hwfc.utils.mytypes import SUPER_POS

eps = 0.001


class Level:
    def __init__(self, map: np.ndarray, num_tiles: int):
        # Just a level wrapper for keeping track of the relevant properties
        self.map = map
        self.num_tiles = num_tiles
        self.extra_freq = None

    def count_pattern(self, pat: "Level", ret_loc=False) -> int:
        size = pat.map.shape
        T = 0
        locs = []
        if not all([t == SUPER_POS.value for t in pat.map.flatten()]):
            for i in range(self.map.shape[0] - size[0] + 1):
                for j in range(self.map.shape[1] - size[1] + 1):
                    slc = self.map[i : i + size[0], j : j + size[1]]
                    idx = pat.map != SUPER_POS.value
                    if np.all(slc[idx] == pat.map[idx]):
                        T += 1
                        locs.append((i, j))
        if ret_loc:
            return T, locs
        return T


DDD = {}


def make_prob_dist_from_2d_level(A: np.ndarray, size=2, total_num_tiles=2):
    # Calculates a probability distribution over tile patterns from a 2d level
    global DDD
    kkk = (A.tostring(), size, total_num_tiles)
    if kkk in DDD:
        return DDD[kkk]
    if len(DDD) > 25000:
        DDD = {} # to not run out of RAM
    # As described in: Tile pattern kl-divergence for analysing and evolving game levels; https://arxiv.org/pdf/1905.05077
    A = A.astype(np.int32)
    N = A.shape[0]
    total = total_num_tiles ** (size ** 2)
    prob_dist = defaultdict(lambda: 0)
    assert A.shape == (N, N)
    num_blocks = N - size + 1

    pp = []
    for s in range(size ** 2):
        pp.append(total_num_tiles ** s)
    pp = np.array(pp)
    for i in range(num_blocks):
        for j in range(num_blocks):
            a, b = i, j
            little_block = A[a : a + size, b : b + size].flatten()
            nums = (little_block * pp).sum()
            prob_dist[nums] += 1
    CC = np.sum(list(prob_dist.values()))
    new = {}
    for k, v in prob_dist.items():
        new[k] = (v + eps) / ((CC + eps) * (1 + eps))
    ans = new, dict(prob_dist)
    DDD[kkk] = ans
    return ans


image_hash_cache = {}


def _general_image_hash_distance(
    which_hash: Callable[[Image.Image, int], imagehash.ImageHash], a: np.ndarray, b: np.ndarray, hash_size=8
) -> float:
    """Performs the image hashing distance, from here: https://github.com/JohannesBuchner/imagehash/blob/master/LICENSE

    Args:
        a (np.ndarray):
        b (np.ndarray):
        which_hash: Callable[[Image.Image, int], imagehash.ImageHash] The image hashing function, could be any of
            imagehash.phash_simple
            imagehash.average_hash
            imagehash.phash
            imagehash.whash
    Returns:
        float: The distance, which is normalised to between 0 and 1 under the assumption that it's capped at 64 = hash_size * hash_size, but I might be wrong.
    """
    if len(a.shape) == 3 and a.shape[1] == 1:
        a = a[:, 0]
    if len(b.shape) == 3 and b.shape[1] == 1:
        b = b[:, 0]

    ka = (a.tostring(), hash_size, which_hash)
    kb = (b.tostring(), hash_size, which_hash)

    a_hash = (
        image_hash_cache[ka]
        if ka in image_hash_cache
        else which_hash(Image.fromarray(a.astype(np.uint8)), hash_size=hash_size)
    )
    b_hash = (
        image_hash_cache[kb]
        if kb in image_hash_cache
        else which_hash(Image.fromarray(b.astype(np.uint8)), hash_size=hash_size)
    )

    if ka not in image_hash_cache:
        image_hash_cache[ka] = a_hash
    if kb not in image_hash_cache:
        image_hash_cache[kb] = b_hash
    return (a_hash - b_hash) / (hash_size ** 2)


# Different functions for image hashing
def image_hash_distance_perceptual_simple(a: np.ndarray, b: np.ndarray) -> float:
    return _general_image_hash_distance(imagehash.phash_simple, a.map, b.map)


def nov_calc_phash(l1: np.ndarray, l2: np.ndarray):
    return image_hash_distance_perceptual_simple(l1, l2)


# Hamming distance
def nov_calc_hamming(l1: np.ndarray, l2: np.ndarray):
    return (l1.map != l2.map).sum() / l1.map.size


def do_kl(pdist, qdist, pcounts):
    # Does the KL divergence calculation
    ss = set(k for k, v in pcounts.items() if v > 0)
    all_keys = list((set(list(pdist.keys()) + list(qdist.keys()))).intersection(ss))
    p = np.zeros(len(all_keys))
    q = np.zeros(len(all_keys))
    assert len(all_keys) > 0
    for i, k in enumerate(all_keys):
        if k in pdist:
            p[i] += pdist[k]
        else:
            p[i] += eps

        if k in qdist:
            q[i] += qdist[k]
        else:
            q[i] += eps
    sums2 = (p * np.log(p / q)).sum()
    return sums2


def avg_kl(pdist, qdist, pcounts, qcounts):
    # This averages KL(p||q) and KL(q||p) because it is not symmetric.
    a = do_kl(pdist, qdist, pcounts)
    b = do_kl(qdist, pdist, qcounts)
    assert a >= 0 and a < np.inf
    assert b >= 0 and b < np.inf
    return (a + b) / 2


def nov_calc_tile_kl_size_general(size, l1: np.ndarray, l2: np.ndarray, numtiles):
    pdist, countsp = make_prob_dist_from_2d_level(l1, size=size, total_num_tiles=numtiles)
    qdist, countsq = make_prob_dist_from_2d_level(l2, size=size, total_num_tiles=numtiles)
    return avg_kl(pdist, qdist, countsp, countsq)


# Different sized KL calculations
def nov_calc_tile_kl_size2(l1: Level, l2: Level):
    return nov_calc_tile_kl_size_general(2, l1.map, l2.map, numtiles=l1.num_tiles)


def nov_calc_tile_kl_size3(l1: Level, l2: Level):
    return nov_calc_tile_kl_size_general(3, l1.map, l2.map, numtiles=l1.num_tiles)


def nov_calc_tile_kl_size4(l1: Level, l2: Level):
    return nov_calc_tile_kl_size_general(4, l1.map, l2.map, numtiles=l1.num_tiles)


def nov_calc_tile_kl_size5(l1: Level, l2: Level):
    return nov_calc_tile_kl_size_general(5, l1.map, l2.map, numtiles=l1.num_tiles)


def nov_calc_tile_kl_size6(l1: Level, l2: Level):
    return nov_calc_tile_kl_size_general(6, l1.map, l2.map, numtiles=l1.num_tiles)
