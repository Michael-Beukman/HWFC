import pickle
from typing import Literal, get_args
import sys

import fire
from hwfc.generators.hierarchical_wfc import HWFCGenerator
from hwfc.generators.multiscale_wfc import MultScaleWFCGenerator

from hwfc.generators.wfc import WFCGenerator
from hwfc.minecraft.mc_utils import seed_minecraft_world


Game   = Literal["maze", "2.5D", "minecraft"]
Method = Literal["HWFC", "WFC", "MWFC"]

def main(game: Game, method: Method, level_size: int = 30, seed: int = 42):
    assert game in get_args(Game)
    assert method in get_args(Method)
    # Get examples:
    LSIZE = (level_size, level_size)
    extra_kwargs = {}
    SIZES = [(2, 2), (3, 3)]
    
    if game == 'maze':
        BASE_EXAMPLES = ['examples/maze/example.txt']
        H_EXAMPLES = [['examples/maze/top_1.txt', 'examples/maze/top_2.txt'], [f'examples/maze/mid_{i}.txt' for i in range(1, 5)]]
        MAX_COUNTS = [1, 0.33, 1.0]
        MAX_COUNTS_PER_PATTERNS = [[1, 1], [1, 1, 1, 1], None]
    elif game == '2.5D':
        TOP_HIERARCHIES = [f'examples/2.5D/top_1/Rot{j}.txt' for j in range(1, 5)] + [f'examples/2.5D/top_2/Rot{j}.txt' for j in range(1, 5)]
        MID_HIERARCHIES = [f'examples/2.5D/mid_{i}.txt' for i in [1, 2]] + [f'examples/2.5D/mid_{i}/Rot{j}.txt' for i in [3, 4] for j in range(1, 5)]
        BASE_EXAMPLES = [f'examples/2.5D/example_1/Rot{i}.txt' for i in range(1,5)] + [f'examples/2.5D/example_2/Rot{i}.txt' for i in range(1, 5)]
        H_EXAMPLES = [TOP_HIERARCHIES, MID_HIERARCHIES]
        MAX_COUNTS = [1, 3, 1.0]
        MAX_COUNTS_PER_PATTERNS = [[1, 1], [1, 1, 1, 1], None]
    elif game == 'minecraft':
        from hwfc.external.minecraft_pb2 import AIR, STONE
        from hwfc.minecraft.place import place_in_mc
        # Minecraft
        LSIZE = (level_size, 7, level_size)
        H_EXAMPLES = [
            ['examples/minecraft/5_16_5_16.txt', 'examples/minecraft/4_15_6_15.txt'],
            ['examples/minecraft/1_7_5_7.txt', 'examples/minecraft/3_7_3_7.txt', 'examples/minecraft/2_4_4_4.txt']
            ]
        BASE_EXAMPLES = ['examples/minecraft/0_25_25_25.txt']

        extra_kwargs = dict(seed_world_func=seed_minecraft_world, max_pat_occurrence=400, check_cropped_done=False)
        SIZES = [(2, 2, 2), (3, 3, 3)]
        MAX_COUNTS = [1, 1, 1.0]
        MAX_COUNTS_PER_PATTERNS = [[1, 1], [1, 1, 1], None]
    else:
        print(f"Received an invalid game: '{game}'")
        exit(1)
    ALL_VALS = BASE_EXAMPLES + sum(H_EXAMPLES, [])
    
    
    common_kwargs = dict(level_size=LSIZE, domain=game, replace_date_with_seed=True, seed=seed, desired_sizes=SIZES, ignore_superpos_tiles=True, plot_correct_aspect_ratio=True) | extra_kwargs
    
    standard_wfc = WFCGenerator(base_examples=ALL_VALS, **common_kwargs)
    
    hwfc = HWFCGenerator(base_examples=BASE_EXAMPLES, hierarchy_examples=H_EXAMPLES, max_counts=MAX_COUNTS, max_counts_per_patterns=MAX_COUNTS_PER_PATTERNS, **common_kwargs)
    
    mwfc = MultScaleWFCGenerator(base_examples=BASE_EXAMPLES, hierarchy_examples=H_EXAMPLES, **common_kwargs)
    
    gen = {
            "MWFC": mwfc,
            "HWFC": hwfc,
            "WFC": standard_wfc
        }[method]

    ans = gen.generate(seed=seed)
    if game == 'minecraft':
        place_in_mc(ans)
    
if __name__ == '__main__':
    fire.Fire(main)