import copy
import math
import os
from typing import Dict, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from hwfc.utils.classes import InputPattern, Tile
from hwfc.utils.conf import IMAGE_COLOUR_MAP
from hwfc.utils.mytypes import SUPER_POS
from hwfc.world import SuperPosition, World
import hwfc.utils.helpers as helpers
sns.set_theme()

def get_img_from_map_or_pattern(map_or_pat: Union[World, InputPattern]) -> np.ndarray:
    """Given a map or a pattern, return an np array representing the map/pattern as an image.

    Args:
        map_or_pat (Union[World, InputPattern]): 

    Returns:
        np.ndarray: 
    """
    def get_img_from_map(map: World):
        if len(map.map.shape) == 3:
            map = copy.deepcopy(map)
            map.map = map.map.reshape((map.map.shape[0], map.map.shape[1] * map.map.shape[2])) # a hack to plot the 3D map as a 2D one
        def get_colour(k: SuperPosition):
            if k.possible_tiles == [SUPER_POS]:
                return -1
            if k.is_collapsed: 
                return k.possible_tiles[0].value
            else: 
                assert len(k.possible_tiles) > 1
                return -1
        return np.array([[get_colour(k) for k in l] for l in map.map])
    
    def get_img_from_pattern(map: InputPattern):
        if len(map.tiles.shape) == 3:
            map = copy.deepcopy(map)
            map.tiles = map.tiles.reshape((map.tiles.shape[0], map.tiles.shape[1] * map.tiles.shape[2]))
        def get_colour(k: Tile):
            if k == SUPER_POS: return -1
            return k.value
        return np.array([[get_colour(k) for k in l] for l in map.tiles])

    
    return get_img_from_map(map_or_pat) if isinstance(map_or_pat, World) else get_img_from_pattern(map_or_pat)


def good_ax(ax):
    # Removes the x and y axis lines
    ax.tick_params(axis='both',  which='both',  bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False) 


def plot_world_or_pattern(map_or_pat: Union[World, InputPattern], save_name: str, colour_dic: Dict[int, str]=IMAGE_COLOUR_MAP, rect: Tuple[int, int]=None, ax: plt.Axes=None, title: str=None, make_empty=False, correct_aspect_ratio: bool = False):
    """This plots a map/pattern and optionally saves it to a file.

    Args:
        map_or_pat (Union[World, InputPattern]): The map/pattern to plot
        save_name (str): The filename to save the image to
        colour_dic (Dict[int, str], optional): A dictionary defining which values go to which colours. Defaults to IMAGE_COLOUR_MAP.
        rect (Tuple[int, int], optional): If Given, plots a rectangle on that square. Defaults to None.
        ax (plt.Axes, optional): If given, plots the image to this axis, otherwise uses plt.gca(). Defaults to None.
        title (str, optional): If given, adds a title to the plot. Defaults to None.
        make_empty (bool, optional): If True, removes the x and y ticks. Defaults to False.
        correct_aspect_ratio (bool, optional): If True, sets the aspect ratio to be correct with regards to the level's dimensions. Defaults to False.
    """
    if save_name:
        if os.sep in save_name:
            filename = save_name
        else:
            filename = os.path.join(f"plots",f"map_{save_name}.png")
        helpers.check_dir_exists(filename)
    else:
        filename = save_name

    if ax is None and correct_aspect_ratio:
        aspect = map_or_pat.shape[1] / map_or_pat.shape[0]
        plt.figure(figsize=(5, 5 * aspect))

    img = get_img_from_map_or_pattern(map_or_pat)
    sns.heatmap(img, cmap=list(colour_dic.values()), vmin=-1, vmax=len(colour_dic)-1, ax=ax, cbar=not make_empty)
    if ax is None: ax = plt.gca()
    if rect is not None:
        import matplotlib.patches as patches
        ax.add_patch(
            patches.Rectangle(
                (rect),
                1.0,
                1.0,
                edgecolor='purple',
                fill=False,
                lw=2
            ) )
        
    if make_empty: good_ax(ax)
    if title is not None: 
        ax.set_title(title)
    if save_name:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0); plt.close()


def get_subplot_size(n_things: int, basefigsize: float = 5, force_size=None) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """Returns a decent size for a set of subplots given the number of elements to plot

    Args:
        n_things (int): _description_

    Returns:
        Tuple[int, int]: _description_
    """
    
    a = int(math.sqrt(n_things))
    b = math.ceil(n_things / a)
    if force_size is not None and force_size:
        a, b = force_size
    return (a, b), (b * basefigsize, a * basefigsize)


def mysubplots(*args, ravel=True, **kwargs):
    """
        Returns subplots, and ravels them if necessary
    """
    fig, axs = plt.subplots(*args, **kwargs)
    if hasattr(axs, '__len__'): 
        if ravel: axs = axs.ravel()
    else: axs = [axs]
    return fig, axs

def mysubplots_directly(n_things: int, basefigsize: float = 5, force_size=None, ravel=True, additional_y_size=0, *args, **kwargs):
    """Effectively combines `get_subplot_size` and `mysubplots`. Returns figure, axis

    Args:
        n_things (int): _description_
        basefigsize (float, optional): _description_. Defaults to 5.
        force_size (_type_, optional): _description_. Defaults to None.
        ravel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    rc, s = get_subplot_size(n_things, basefigsize, force_size)
    if additional_y_size > 0:
        s = (s[0] + additional_y_size, s[1])
    return mysubplots(*rc, figsize=s, ravel=ravel, *args, **kwargs)
    
