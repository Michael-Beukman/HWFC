import sys
from hwfc.generators.hierarchical_wfc import HWFCGenerator
from hwfc.generators.multiscale_wfc import MultScaleWFCGenerator

from hwfc.generators.wfc import WFCGenerator

def main(name: str, start_val: int, total: int):
    # See auto.sh for usage. T runs all of our main quantitative experiments.
    global DOMAIN
    N = 320
    per_set = N // total
    for seed in range(per_set * start_val, per_set * (start_val + 1)):
        LSIZE = (30, 30)
        if DOMAIN == 'maze':
            BASE_EXAMPLES = ['examples/maze/example.txt']
            H_EXAMPLES = [['examples/maze/top_1.txt', 'examples/maze/top_2.txt'], [f'examples/maze/mid_{i}.txt' for i in range(1, 5)]]
            MAX_COUNTS = [1, 0.33, 1.0]
            MAX_COUNTS_PER_PATTERNS = [[1, 1], [1, 1, 1, 1], None]
        else:
            TOP_HIERARCHIES = [f'examples/2.5D/top_1/Rot{j}.txt' for j in range(1, 5)] + [f'examples/2.5D/top_2/Rot{j}.txt' for j in range(1, 5)]
            MID_HIERARCHIES = [f'examples/2.5D/mid_{i}.txt' for i in [1, 2]] + [f'examples/2.5D/mid_{i}/Rot{j}.txt' for i in [3, 4] for j in range(1, 5)]
            BASE_EXAMPLES = [f'examples/2.5D/example_1/Rot{i}.txt' for i in range(1,5)] + [f'examples/2.5D/example_2/Rot{i}.txt' for i in range(1, 5)]
            H_EXAMPLES = [TOP_HIERARCHIES, MID_HIERARCHIES]
            MAX_COUNTS = [1, 3, 1.0]
            MAX_COUNTS_PER_PATTERNS = [[1, 1], [1, 1, 1, 1], None]
        
        ALL_VALS = BASE_EXAMPLES + sum(H_EXAMPLES, [])
        SIZES = [(2,2), (3,3)]

        standard_wfc = WFCGenerator(level_size=LSIZE, domain=DOMAIN, base_examples=ALL_VALS, replace_date_with_seed=True, seed=seed, desired_sizes=SIZES, ignore_superpos_tiles=True)
        hwfc = HWFCGenerator(level_size=LSIZE, domain=DOMAIN, base_examples=BASE_EXAMPLES, hierarchy_examples=H_EXAMPLES, replace_date_with_seed=True, seed=seed, desired_sizes=SIZES, max_counts=MAX_COUNTS, max_counts_per_patterns=MAX_COUNTS_PER_PATTERNS, ignore_superpos_tiles=True)
        
        
        mwfc = MultScaleWFCGenerator(level_size=LSIZE, domain=DOMAIN, base_examples=BASE_EXAMPLES, hierarchy_examples=H_EXAMPLES, replace_date_with_seed=True, seed=seed, desired_sizes=SIZES, ignore_superpos_tiles=True)

        if name == 'MWFC':
            method = mwfc
        elif name == 'HWFC':
            method = hwfc
        else:
            method = standard_wfc
        if not method.already_exists(): method.generate(seed=seed)
        else: print("Method ", method, 'already exists with seed', seed)


if __name__ == '__main__':
    DOMAIN = sys.argv[-4]
    main(str(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1]))
