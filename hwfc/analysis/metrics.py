import glob
import json
import pathlib
from collections import defaultdict
from hashlib import md5
from typing import Dict, List

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted

from hwfc.analysis.novs import (Level, nov_calc_hamming, nov_calc_phash,
                                nov_calc_tile_kl_size2, nov_calc_tile_kl_size3,
                                nov_calc_tile_kl_size4, nov_calc_tile_kl_size5,
                                nov_calc_tile_kl_size6)
from hwfc.analysis.utils import get_all_subpatterns
from hwfc.utils.classes import InputPattern, Tile
from hwfc.utils.helpers import load_example_from_file
from hwfc.utils.mytypes import SUPER_POS, TOT_TILES
from hwfc.utils.viz import plot_world_or_pattern

PLOT_LEVELS = False

# Calculates the various metrics.

def get_appropriate_variables_for_game(game='maze', do_ex=False):
    # Returns the appropriate values for each game, e.g., which patterns, etc.
    ## MAZE
    NUM_TILES = 2
    TOP_PATTERNS = ['examples/maze/top_1.txt', 'examples/maze/top_2.txt']
    MID_PATTERNS = [f'examples/maze/mid_{i}.txt' for i in range(1, 5)]
    EXAMPLES = ['examples/maze/example.txt']
    MAX_COUNT = 100
    if game == 'maze':
        if do_ex:
            return NUM_TILES, TOP_PATTERNS, MID_PATTERNS, MAX_COUNT, EXAMPLES
        return NUM_TILES, TOP_PATTERNS, MID_PATTERNS, MAX_COUNT
    
    # 2.5D
    NUM_TILES = 50
    TOP_PATTERNS = [f'examples/2.5D/top_1/Rot{j}.txt' for j in range(1, 5)] + [f'examples/2.5D/top_2/Rot{j}.txt' for j in range(1, 5)]
    MID_PATTERNS = [f'examples/2.5D/mid_{i}.txt' for i in [1, 2]] + [f'examples/2.5D/mid_{i}/Rot{j}.txt' for i in [3, 4] for j in range(1, 5)]
    EXAMPLES = [f'examples/2.5D/example_1/Rot{i}.txt' for i in range(1,5)] + [f'examples/2.5D/example_2/Rot{i}.txt' for i in range(1, 5)]
    if game == '2.5D':
        if do_ex:
            return NUM_TILES, TOP_PATTERNS, MID_PATTERNS, MAX_COUNT, EXAMPLES
        return NUM_TILES, TOP_PATTERNS, MID_PATTERNS, MAX_COUNT

def novelty_metric(levels: List[Level], nov_func) -> float:
    scores = np.zeros((len(levels), len(levels)))
    for i, l1 in enumerate(levels):
        for j, l2 in enumerate(levels):
            if i == j:
                continue
            scores[i, j] = nov_func(l1, l2)
    a = np.ones_like(scores, dtype=np.bool8)
    ii = np.arange(len(a))
    a[ii, ii] = 0
    return scores[a].tolist()

def make_pat_level(pat: InputPattern, num_tiles) -> Level:
    test = [[c.value for c in l] for l in pat.tiles]
    return Level(np.array(test), num_tiles=num_tiles)


def get_counts(tt):
    level, all_checks = tt
    all_counts = {k: 0 for k in all_checks}
    for k, v in all_checks.items():
        for pat in v:
            all_counts[k] += level.count_pattern(pat)
    return all_counts


def run_para2(func, itera, procs=8):
    from multiprocessing import Pool
    with Pool() as mp_pool:
        ans = mp_pool.map(func, itera)
    
    return ans

def structure_metric(levels: Dict[str, Level]):
    ALL_PATS = [load_example_from_file(f) for f in TOP_PATTERNS]
    all_checks = defaultdict(lambda: [])
    for desired_pattern, f in zip(ALL_PATS, TOP_PATTERNS):
        f = '_'.join(f.split('/')[-2:])
        s = min(desired_pattern.shape)
        for i in range(2, s + 1): all_checks[str((i, i)) + "__" + f].extend([make_pat_level(p, num_tiles=NUM_TILES) for p in get_all_subpatterns(desired_pattern, size=(i, i))])
        if desired_pattern.shape != (s, s):
            all_checks[str(desired_pattern.shape) + "__" + f].extend([make_pat_level(desired_pattern, NUM_TILES)])
    
    all_checks = dict(all_checks)
    ans = {}
    # Now check occurrences of each pattern, try do parallel
    for method, ls in tqdm.tqdm(levels.items(), total=len(levels)):
        all_outputs = run_para2(get_counts, [(level, all_checks) for level in ls])
        all_counts = {k: 0 for k in all_checks}
        for d in all_outputs:
            for k, v in d.items():
                all_counts[k] += v
        ans[method] = all_counts
    return ans


def get_superpos_rectangle(pattern: Level):
    idx = pattern.map == SUPER_POS.value 
    first_idx = np.unravel_index(np.argmax(idx), idx.shape)
    last_idx = np.unravel_index(idx.size - np.argmax(idx[::-1, ::-1]) - 1, idx.shape)
    value = (pattern.map[first_idx[0]: last_idx[0] + 1, first_idx[1]: last_idx[1] + 1])
    
    return first_idx[0], last_idx[0] + 1, first_idx[1], last_idx[1] + 1

def control_metric(levels: Dict[str, Level]):
    # Get hierarchy 0 and 1
    # Then get the number of times pats from level 1 are placed *within* level 0
    pats_top = [make_pat_level(load_example_from_file(f), NUM_TILES) for f in TOP_PATTERNS]
    pats_mid = [make_pat_level(load_example_from_file(f), NUM_TILES) for f in MID_PATTERNS]
    
    rects_top = [get_superpos_rectangle(p) for p in pats_top]
    ans = {}
    for method, ls in levels.items():
        INSIDE = defaultdict(lambda: 0)
        INSIDEc = 0
        ALL    = defaultdict(lambda: 0)
        ALLc    = 0
        BAD    = 0
        for level in tqdm.tqdm(ls, desc=method):
            for fname, mid_pat in zip(MID_PATTERNS, pats_mid):
                c, l = level.count_pattern(mid_pat, ret_loc=True)
                ALLc += c
                ALL[fname] += c
            all_locs = []
            is_bad = True
            for iii, top in enumerate(pats_top):
                a, b, c, d = rects_top[iii]
                counts, locs = level.count_pattern(top, ret_loc=True)
                if counts != 0:
                    is_bad = False
                all_locs += locs
                for (i, j) in locs:
                    slice = level.map[i + a: i + a + b, j + c: j + c + d]
                    for fname, mid_pat in zip(MID_PATTERNS, pats_mid):
                        X = Level(slice, NUM_TILES).count_pattern(mid_pat)
                        INSIDEc += X
                        INSIDE[fname] += X
            if is_bad:
                BAD += 1
        ans[method] = {'inside': INSIDEc, 'bad': BAD, 'all': ALLc, 'all_d': dict(ALL), 'inside_d': dict(INSIDE)}
    return ans


def get_action_variance(filename):
    C = defaultdict(lambda: 0)
    all_pats = glob.glob(f'{filename}/texts/steps/*/*selected*pat_(*).txt')
    test = glob.glob(f'{filename}/texts/patterns/*/*.txt')
    test = natsorted(test)
    num = int(test[-1].replace(".txt", "").split('/')[-1])
    for p in all_pats:
        t = pathlib.Path(p).read_text().strip()
        t = (md5(t.encode('utf-8')).hexdigest() + "__" + t)
        C[t] += 1
    for c in range(num + 1 - len(C)):
        if c not in C: C[c] = 0
    return dict(C)


def get_methods_and_levels(game='maze', max_count=10):
    numtiles = NUM_TILES
    np.random.seed(42)
    TT = Level((np.random.rand(28, 28) > 0.5) * 1, numtiles)
    def get_from_single_method(name='Hierarchical_WFC'):
        if name == 'Same':
            return [TT for _ in range(max_count)]
        if name == 'Random':
            ss = (28, 28) if CROPPED else (30, 30)
            return [
                Level(np.random.randint(0, numtiles, size=ss), numtiles) for _ in range(max_count)
            ]
        levels = []
        all_vs = []
        fnames = []

        fpath = f'all_exps/main/{game}/{name}/*'
        for seed in natsorted(glob.glob(fpath)):
            try:
                if CROPPED:
                    vv = load_example_from_file(f'{seed}/texts/final_level_cropped.txt')
                else:
                    vv = load_example_from_file(f'{seed}/texts/final_level.txt')    
                    mtest = [[c.value for c in l[1:-1]] for l in vv.tiles[1:-1]]
                    tt = sum(mtest, [])
                    if tt.count(49) > 0: continue
                    vv.tiles = np.array([[c if c != SUPER_POS else Tile(0, TOT_TILES) for c in r] for r in vv.tiles])
            except Exception as e:
                print(e)
                continue
            test = [[c.value for c in l] for l in vv.tiles]
            tt = sum(test, [])
            if CROPPED:
                if tt.count(49) > 0: continue
            else:
                mtest = [[c.value for c in l[1:-1]] for l in vv.tiles[1:-1]]
                tt = sum(mtest, [])
                if tt.count(49) > 0: continue
            map = np.array(test)
            
            levels.append(Level(map, num_tiles=numtiles))
            all_vs.append(vv)
            fnames.append(seed)
            if len(levels) >= max_count: break
        
        for l, ff in zip(levels, fnames):
            l.extra_freq = get_action_variance(ff)
        if PLOT_LEVELS:
            def save(game, name):
                path = f'results/paper/metrics/levels/{game}/{CROPPED}/{name}.png'
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()
            fig, axs = plt.subplots(10, 10, figsize=(30, 30))
            for i, (p, a) in enumerate(zip(all_vs, axs.ravel())):
                print(i)
                plot_world_or_pattern(p, ax=a, save_name=None, make_empty=True)
                a.set_title(fnames[i].split('/')[-1])
            save(game, f'{name}')
        return levels
    return {k: get_from_single_method(k) for k in ['Normal_WFC', 'Hierarchical_WFC', 'Multiscale_WFC', 'Random', "Same"][:]}

def plot_structure(structure):
    alls = {}
    for method, v in structure.items():
        x, y = [], [] 
        for size, counts in v.items():
            x.append(size[0] * size[1])
            y.append(counts)
        alls[method] = (x, y)
    # normalise
    norm_y = []
    for i in range(len(x)):
        v = -1000
        v = 0
        for k in alls:
            v = max(v, alls[k][1][i])
        norm_y.append(v)
        
    for method, (x, y) in alls.items():
        Y = np.array(y) / np.array(norm_y)
        plt.plot(x, Y, label=method)
    plt.legend()
    path = 'results/paper/plots/structure/structure.png'
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); plt.close()

def get_novelty_subpattern_levels(levels, nov_func):
    test = []
    for level in levels:
        # Split this one up into 4
        A, B = level.map.shape
        A = A // 2
        B = B // 2
        
        
        x = []
        x.append(Level(level.map[:A, :B], level.num_tiles))
        x.append(Level(level.map[A:, :B], level.num_tiles))
        x.append(Level(level.map[:A, B:], level.num_tiles))
        x.append(Level(level.map[A:, B:], level.num_tiles))
        
        test.append(novelty_metric(x, nov_func))
    return test
        
def main(game):
    NOVS = {
        'phash': nov_calc_phash, 
        'hamming': nov_calc_hamming, 
        'kl2': nov_calc_tile_kl_size2, 
        'kl3': nov_calc_tile_kl_size3,
        'kl4': nov_calc_tile_kl_size4,
        'kl5': nov_calc_tile_kl_size5,
        'kl6': nov_calc_tile_kl_size6,
    }
    levels = get_methods_and_levels(game, max_count=MAX_COUNT)
    metrics = {
        k: {
            'novelty': {},
            'novelty_inner': {},
            'structure': 0,
            'control': 0,
            'extra_action_variance': []
            
        }
        for k in levels.keys()
    }
    for k, v in metrics.items():
        v['extra_action_variance'] = [level.extra_freq for level in levels[k]]
    if CROPPED:
        for k, v in levels.items():
            for nk, nv in NOVS.items():
                metrics[k]['novelty'][nk]       = novelty_metric(v, nv)
                # Get subpattern levels
                metrics[k]['novelty_inner'][nk] = get_novelty_subpattern_levels(v, nv)

    control_vals    = control_metric(levels)
    for method, v in control_vals.items():
        metrics[method]['control'] = v
    
    structure_vals  = structure_metric(levels)
    for method, v in structure_vals.items():        
        metrics[method]['structure'] = v
    path = f'results/paper/plots/all/{game}/data_Cropped_{CROPPED}.json'
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    for CROPPED in [True, False]:
        for game in ['maze', '2.5D']:
            NUM_TILES, TOP_PATTERNS, MID_PATTERNS, MAX_COUNT = get_appropriate_variables_for_game(game)
            print(game)
            main(game)
            
