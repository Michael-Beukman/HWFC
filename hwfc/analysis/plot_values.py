from collections import defaultdict
from email.policy import default
import glob
from hashlib import md5
import json
import pathlib

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
def start():
    plt.figure(figsize=(10, 6))

def clean_method(m):
    a = m.replace("_WFC", '')
    if a == 'Normal': return 'WFC'
    if a == 'Hierarchical': return 'HWFC'
    if a == 'Multiscale': return 'MWFC'
    if a == 'Multiscale': return 'MWFC'
    return a
def clean_game(s):
    if s == 'maze': return 'Maze'
    elif s == '2.5D': return 'Rogue'
    else: assert False

PRETTY_LINEWIDTH_PLOT = 3.5
PRETTY_LINEWIDTH = 6

def make_legend_good(leg, skiplast=False, skipn=0):
    if True:
        ll = leg.legendHandles
        if skiplast:
            ll = leg.legendHandles[:-1]
        if skipn > 0:
            ll = leg.legendHandles[:-skipn]
        for legobj in ll: legobj.set_linewidth(PRETTY_LINEWIDTH)

def savefig(name, game_override=None):
    if game_override is None: game_override = GAME
    path = f'results/paper/plots/all/{game_override}/{name}.png'
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight', pad_inches=0); 
    plt.savefig(path.replace("png", "pdf"), bbox_inches='tight', pad_inches=0); 
    
    plt.close()

def single_structure_plot(structure, mode, ax, labels=True):
    alls = {}
    overall = defaultdict(lambda: defaultdict(dict))
    for method, v in structure.items():
        if 'Random' in method or 'Same' in method: continue
        tests = defaultdict(lambda: 0)
        x, y = [], [] 
        for size, counts in v.items():
            pat_name = size.split("__")[1]
            size = tuple(map(lambda x: int(x.split("__")[0]), size.replace('(', '').replace(')', '').split(", ")))
            tests[size[0] * size[1]] += counts
            overall[method][pat_name][size] = counts
        for size, counts in sorted(tests.items()):
            x.append(size)
            y.append(counts)
        alls[method] = (x, y)
    norm_y = []
    for i in range(len(x)):
        v = -1000
        v = 0
        for k in alls:
            v += (alls[k][1][i])
        norm_y.append(v)


    if mode == 'all':
        ttt = 0
        width = 2
        x_vals = np.arange(len(x)) * (width*3 + 2)
        for method, (x, y) in alls.items():
            Y = np.array(y) 
            x = list(map(str, x))
            ax.bar(x_vals + ttt, Y, label=clean_method(method), width=width)
            ttt += width
        
        ax.set_xticks(x_vals + width, x)
        if labels:
            ax.set_yscale('log')
            ax.set_ylabel("Fraction of Occurrences")
            ax.set_xlabel("Size of Pattern")
            ax.legend()
        return
    elif mode == 'only':
        x, y = [], []
        for method, v in overall.items():
            x.append(clean_method(method))
            tot = 0
            for k, vv in v.items():
                kk = max(vv.keys(), key=lambda x: x[0] * x[1])
                tot += vv[kk]
            y.append(tot)
        ax.bar(x, y)
        if labels:
            ax.set_xlabel("Method")
            ax.set_ylabel("Number of Occurrences of large pattern")
        return


def single_novelty_plot(novelty, metric, ax, labels=True):
    x, y = [], []
    stds = []
    for k, v in novelty.items():
        if 'Same' in k: continue
        stds.append(np.std(v[metric]))
        y.append(np.mean(v[metric]))
        x.append(clean_method(k))
    ax.bar(x, y, yerr=stds, capsize=5)
    if labels:
        ax.set_xlabel("Method")
        ax.set_ylabel("Novelty Score")

def plot_novelty_kl_sizes(novelty, ax):
    DD = defaultdict(lambda: ([], [], []))
    for p in range(2, 7):
        metric = f'kl{p}'
        for k, v in novelty.items():
            if 'Same' in k: continue
            DD[clean_method(k)][0].append(p)
            DD[clean_method(k)][1].append(np.mean(v[metric]))
            DD[clean_method(k)][2].append(np.std(v[metric]))
    
    for k, v in DD.items():
        v = [np.array(x) for x in v]
        ax.plot(v[0], v[1], label=k, lw=PRETTY_LINEWIDTH_PLOT)

def single_control_plot(control, norm, proper_name, ax, labels=True):
    x, y1, y2 = [], [], []
    for method, value in control.items():
        if 'Random' in method or 'Same' in method: continue
        if proper_name == '':
            i = value['inside']
            a = value['all']
        else:
            i = value['inside_d'].get(proper_name, 0)
            a = value['all_d'].get(proper_name, 0)
            
        x.append(clean_method(method))
        y1.append(i)
        y2.append(a - i)

    y1 = np.array(y1, dtype=np.float32)
    y2 = np.array(y2, dtype=np.float32)

    s = y1 + y2
    i = s != 0

    if norm:
        y1[i] = y1[i] / s[i]
        y2[i] = y2[i] / s[i]
    p = ax.bar(x, y1,            label='Inside big hierarchy')
    p = ax.bar(x, y2, bottom=y1, label='Outside big hierarchy')
    if labels:
        ax.legend()
        ax.set_xlabel("Method")
        ax.set_ylabel("Fraction of Occurrences")

def plot_intra_action_variance(datas):
    def inner(do_only_middle):
        X = [{method: v['extra_action_variance'] for method, v in data.items()} for data in datas]
        
        def isgood(method):
            return 'Hierarchical' in method or 'Multiscale' in method
        # Get a full set of keys
        games = ['maze', '2.5D']
        myans = {}
        for d, game in zip(X, games):
            if game == 'maze': continue
            # Get proper set of keys:
            proper_keys = set()
            for pat in glob.glob(f'all_exps/main/{game}/Hierarchical_WFC/0000/texts/patterns/*/*.txt'):
                t = pathlib.Path(pat).read_text().strip()
                proper_keys.add((md5(t.encode('utf-8')).hexdigest() + "__" + t))
            if game == '2.5D':
                proper_keys |= {'601fef1d495950547a1f3d09fe6d5c36__0,0,18,0,0,\n0,15,18,20,0,\n0,15,18,20,0,\n0,15,18,20,0,\n0,0,18,0,0,', '2764135ad826fff83efb1a2820fa283c__0,0,0,0,0,18,0,0,0,0,18,0,17,0,0,0,0,0,0,0,\n0,0,0,0,0,18,0,29,19,19,27,0,17,0,0,0,0,0,0,0,\n0,0,0,0,0,18,0,18,0,0,0,0,17,0,0,0,0,0,0,0,\n16,16,16,16,16,1,16,1,16,16,16,16,31,0,0,0,0,0,0,0,\n0,0,0,0,0,18,0,18,0,0,0,0,0,0,0,0,0,0,0,0,\n0,6,23,23,23,13,23,13,23,23,23,23,23,23,23,23,23,23,5,0,\n0,24,6,21,21,11,21,11,21,21,21,21,21,21,21,21,21,5,22,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,14,12,19,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,24,22,23,23,23,23,23,23,23,23,23,23,23,23,23,23,4,22,0,\n0,3,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,4,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,', '25780ce4ded65a6ad07379a5a412a442__0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,6,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,5,0,\n0,24,6,21,21,21,21,21,21,21,21,21,21,21,21,21,21,24,22,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n19,14,12,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,49,49,49,49,49,49,49,49,49,49,49,49,0,24,22,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,24,3,23,23,23,23,23,23,23,23,23,13,23,13,23,23,4,22,0,\n0,3,21,21,21,21,21,21,21,21,21,21,11,21,11,21,21,21,4,0,\n0,0,0,0,0,0,0,0,0,0,0,0,18,0,18,0,0,0,0,0,\n0,0,0,0,0,0,0,33,16,16,16,16,1,16,1,16,16,16,16,16,\n0,0,0,0,0,0,0,17,0,0,0,0,18,0,18,0,0,0,0,0,\n0,0,0,0,0,0,0,17,0,29,19,19,27,0,18,0,0,0,0,0,\n0,0,0,0,0,0,0,17,0,18,0,0,0,0,18,0,0,0,0,0,', 'f57cbda855c61f45074c7c2b01bde45c__0,0,0,0,18,0,0,0,0,0,18,0,0,0,0,0,\n0,33,16,16,1,16,16,16,16,16,1,16,16,16,32,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,2,19,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,30,16,16,16,16,16,16,16,16,16,16,16,16,31,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,', '34fc723eb466c7bc76b7bdcf34157fd2__0,0,18,0,0,\n0,20,18,15,0,\n0,20,18,15,0,\n0,20,18,15,0,\n0,0,18,0,0,', 'e9a356fbc6a7c199f348c2837fe187d6__0,0,0,0,0,\n0,20,20,20,0,\n19,19,19,19,19,\n0,15,15,15,0,\n0,0,0,0,0,', '404a51c6d8189182b0e953c2bcb83c1c__0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,33,16,16,16,16,16,16,16,16,16,16,16,16,32,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,2,19,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,2,19,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,30,16,16,1,16,16,16,16,16,16,16,16,16,31,0,\n0,0,0,0,18,0,0,0,0,0,0,0,0,0,0,0,', '4cc71f1d20d34079e8e8144d3b85ca3a__6,23,13,23,5,\n24,25,25,25,22,\n24,25,20,25,22,\n24,25,25,25,22,\n3,21,21,21,4,', '817fafb053a16f26b50998f2b0c94c31__0,0,0,0,0,\n0,15,15,15,0,\n19,19,19,19,19,\n0,20,20,20,0,\n0,0,0,0,0,', '08e3f3a6476c3622d69051617a883138__6,23,23,23,5,\n24,25,25,25,22,\n24,25,20,25,12,\n24,25,25,25,22,\n3,21,21,21,4,', 'f2e5fb6d86545fca426925a1cc00b16f__0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,\n0,33,16,16,16,16,16,16,16,16,16,1,16,16,32,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n19,2,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n19,2,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,30,16,16,16,16,16,16,16,16,16,16,16,16,31,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,', '3afe8bc08c22d965313d7df3e1cf2b1c__0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,33,16,16,16,16,16,16,16,16,16,16,16,16,32,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n19,2,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,17,49,49,49,49,49,49,49,49,49,49,49,49,17,0,\n0,30,16,16,16,1,16,16,16,16,16,1,16,16,31,0,\n0,0,0,0,0,18,0,0,0,0,0,18,0,0,0,0,', '2ab68a43d81509d72afff4dade8bf2c6__6,23,23,23,5,\n24,25,25,25,22,\n24,25,20,25,22,\n24,25,25,25,22,\n3,21,11,21,4,', '6e69f4b4916ba27c5246172a01b5a504__0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,0,\n0,0,0,0,0,6,23,23,23,23,23,13,23,23,23,23,23,23,5,0,\n0,0,0,0,0,24,6,21,21,21,21,11,21,21,21,21,21,5,22,0,\n0,0,0,0,0,24,22,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,0,0,0,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,0,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,0,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n16,16,16,32,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n19,28,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,18,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,18,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,26,19,2,19,14,12,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n19,19,19,2,19,14,12,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,17,0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,\n0,0,0,17,0,24,22,0,0,0,0,0,0,0,0,0,0,24,22,0,\n0,0,0,17,0,24,3,23,23,23,23,23,23,23,23,23,23,23,22,0,\n0,0,0,17,0,3,21,21,21,21,21,21,21,21,21,21,21,21,4,0,\n0,0,0,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,', 'bf858c8e7821b5a1045fa97a55c6679e__0,0,18,0,0,\n0,29,8,28,0,\n19,9,0,7,19,\n0,26,10,27,0,\n0,0,18,0,0,', 'c8487be29c9d60fb7c6a894acc898c7e__6,23,23,23,5,\n24,25,25,25,22,\n14,25,20,25,22,\n24,25,25,25,22,\n3,21,21,21,4,', '0964c392d9ad8b52d4c1c3731bea2a52__0,0,0,0,0,\n0,6,23,5,0,\n0,24,25,22,0,\n0,3,21,4,0,\n0,0,0,0,0,', '53c72494d06f11409ec5181a4bf954ec__0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,0,0,0,\n0,6,23,23,23,23,23,23,23,23,23,23,23,23,5,0,17,0,0,0,\n0,24,21,21,21,21,21,21,21,21,21,21,21,5,22,0,17,0,0,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,24,22,0,17,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,14,12,19,2,19,19,19,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,14,12,19,2,19,28,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,18,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,18,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,26,19,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,17,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,30,16,16,16,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,0,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,0,0,0,0,\n0,24,22,0,49,49,49,49,49,49,49,49,0,24,22,0,0,0,0,0,\n0,24,22,0,0,0,0,0,0,0,0,0,0,24,22,0,0,0,0,0,\n0,24,3,23,23,23,23,23,13,23,23,23,23,4,22,0,0,0,0,0,\n0,3,21,21,21,21,21,21,11,21,21,21,21,21,4,0,0,0,0,0,\n0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,0,0,0,0,'}
            for method, values in d.items():
                if not isgood(method): continue
                all = defaultdict(lambda: 0)
                for x in values: 
                    for kk, vv in x.items(): 
                        if kk.isnumeric(): 
                            assert vv == 0
                            continue
                        all[kk] += vv
                def size(kkk):
                    aa = kkk.split("__")[-1]
                    ans =  aa.count(',')
                    if do_only_middle and ans != 25: ans = -1
                    return ans
                diff = set(all.keys()) - proper_keys
                
                
                kks = [k for k in proper_keys if size(k) != -1]
                maxk = kks[0]
                for myk in kks:
                    if all[myk] > all[maxk]: maxk = myk
                    
                print('==============================')
                print(method, game)
                print(sorted(all.items(), key=lambda x: x[1], reverse=True))
                print('==============================')
                print(method, 'maxk', maxk, all[maxk])
                
                vvv = np.array([all[k] * size(k) for k in kks])
                vvv = vvv / (dd := vvv.sum())
                entropy = -(vvv * np.log2(vvv)).sum()
                entropy = np.std(vvv * dd)
                myans[method] = entropy

        return myans
        
    mid = inner(do_only_middle=True)
    all = inner(do_only_middle=False)
    
    keys = ['HWFC', 'MWFC']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    values = [mid['Hierarchical_WFC'], mid['Multiscale_WFC'],
              all['Hierarchical_WFC'], all['Multiscale_WFC']]
    BLUE, ORANGE = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725),
    BLUE, ORANGE = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    A = axs[0].bar(keys[:2], values[:2])
    B = axs[1].bar(keys[:2], values[2:])
    
    A[1].set_color(BLUE)
    A[0].set_color(ORANGE)
    B[1].set_color(BLUE)
    B[0].set_color(ORANGE)
    axs[0].set_title("Medium-Level")
    axs[1].set_title("All")
    savefig(f'action_variance/all/variance', game_override='2.5D')
    print(mid, all)

def plot_combined_plots():
    FIGSIZE = (10, 3.5)
    games = ['maze', '2.5D']
    datas = [json.loads(pathlib.Path(f'results/paper/plots/all/{game}/data_Cropped_False.json').read_text()) for game in games]
    controls = [{method: v['control'] for method, v in data.items()} for data in datas]
    structures = [{method: v['structure'] for method, v in data.items()} for data in datas]
    
    datas = [json.loads(pathlib.Path(f'results/paper/plots/all/{game}/data_Cropped_True.json').read_text()) for game in games]
    novelties = [{method: v['novelty'] for method, v in data.items()} for data in datas]
    novelties_intra = [{method: v['novelty_inner'] for method, v in data.items()} for data in datas]
    plot_intra_action_variance(datas);
    
    # NOVELTY Hamming
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.95))
    for i, (nov, ax) in enumerate(zip(novelties, axs)):
        single_novelty_plot(nov, 'hamming', ax)
        if i == 0: ax.set_ylabel("Diversity")
            
        ax.set_title(clean_game(games[i]))
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    savefig(f'novelty/all/novelty_combined', game_override='combined')
    
    
    
    # NOVELTY
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    for i, (nov, ax) in enumerate(zip(novelties, axs)):
        plot_novelty_kl_sizes(nov, ax)
        if i == 0: ax.set_ylabel("Diversity")
            
        ax.set_title(clean_game(games[i]))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=5, fontsize=13)
    make_legend_good(leg)
    plt.tight_layout()
    savefig(f'novelty/all/novelty_combined_kl_sizes', game_override='combined')
    
    # Inter NOVELTY
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    for i, (nov, ax) in enumerate(zip(novelties_intra, axs)):
        for a, b in nov.items():
            for c, d in b.items():
                print("len", len(nov[a][c]), a, c)
                assert len(nov[a][c]) == 100
                nov[a][c] = np.mean(nov[a][c])
        plot_novelty_kl_sizes(nov, ax)
        if i == 0: ax.set_ylabel("Diversity")
            
        ax.set_title(clean_game(games[i]))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=5, fontsize=13)
    make_legend_good(leg)
    plt.tight_layout()
    savefig(f'novelty/all/novelty_combined_kl_sizes_inner', game_override='combined')

    # CONTROL
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    for i, (control, ax) in enumerate(zip(controls, axs)):
        single_control_plot(control, True, '', ax, labels=False)
        if i == 0:
            ax.set_ylabel("Fraction of Occurrences")
            
        ax.set_title(clean_game(games[i]))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=5, fontsize=13)
    plt.tight_layout()
    savefig(f'control/all/control_combined', game_override='combined')

    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)
    # Structure
    for i, (structure, ax) in enumerate(zip(structures, axs)):
        single_structure_plot(structure, 'only', ax, labels=False)
        if i == 0:
            ax.set_ylabel("Number of Occurrences")
            
        ax.set_title(clean_game(games[i]))
    plt.tight_layout()
    savefig(f'structure/all/structure_combined', game_override='combined')

    fig, axs = plt.subplots(1, 2, figsize=(15, 3.5))
    # Structure
    for i, (structure, ax) in enumerate(zip(structures, axs)):
        single_structure_plot(structure, 'all', ax, labels=False)
        if i == 0:
            ax.set_ylabel("Number of Occurrences")
        ax.set_yscale('log')
        ax.set_title(clean_game(games[i]))
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=True, shadow=True, ncol=5, fontsize=13)
    plt.tight_layout()
    savefig(f'structure/all/structure_combined_all', game_override='combined')

if __name__ == '__main__':
    plot_combined_plots()