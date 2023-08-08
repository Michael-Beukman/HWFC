
import sys
from hwfc.external.minecraft_pb2 import AIR, STONE
from hwfc.generators.hierarchical_wfc import HWFCGenerator
import pickle
import numpy as np
from hwfc.minecraft.mc_utils import seed_minecraft_world

from hwfc.minecraft.place import place_in_mc


def generate_mc(i):
    level_size = (40, 7, 40)
    # Generate similar levels to the ones in the paper
    I = i // 3
    i = i % 3
    if i == 0:
        tile_sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
        max_counts = [2, 2, 1.0]
        seed = 433 + I
        max_counts_per_patterns=[[5, 5], [1, 1, 1], None]
    elif i == 1:
        tile_sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
        max_counts = [3, 3, 1.0]
        seed = 433 + I
        max_counts_per_patterns=[[5, 5], [1, 1, 1], None]
    elif i == 2:
        tile_sizes = [(2, 2, 2), (3, 3, 3)]
        max_counts = [2, 3, 1.0]
        seed = 633 + I
        max_counts_per_patterns=[[1, 1], [0, 2, 2], None]

    H_EXAMPLES = [
        ['examples/minecraft/5_16_5_16.txt', 'examples/minecraft/4_15_6_15.txt'],
        ['examples/minecraft/1_7_5_7.txt', 'examples/minecraft/3_7_3_7.txt', 'examples/minecraft/2_4_4_4.txt']
    ]
    BASE = ['examples/minecraft/0_25_25_25.txt']

    hwfc = HWFCGenerator(level_size=level_size, domain='minecraft', base_examples=BASE, desired_sizes=tile_sizes, check_cropped_done=False, max_counts=max_counts, hierarchy_examples=H_EXAMPLES, seed=seed, ignore_superpos_tiles=True, max_pat_occurrence=400, seed_world_func=seed_minecraft_world, max_counts_per_patterns=max_counts_per_patterns)


    ans = hwfc.generate(seed=seed)
    place_in_mc(ans)




if __name__ == '__main__':
    generate_mc(int(sys.argv[-1]))