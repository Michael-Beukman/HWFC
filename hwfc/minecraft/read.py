import pickle

from external.minecraft_pb2 import *
from hwfc.minecraft.mc_utils import get_channel_and_client, get_pattern_from_minecraft

import hwfc.utils.helpers as helpers
from hwfc.utils.classes import InputPattern

channel = None
client = None

def read_all_patterns():
    # Reads all of the patterns from these locations and saves them as text files
    LOCS = [
        dict(sx=1000, sz=1027, SIZE=25, size_y=25),
        dict(sx=1000, sz=1000, SIZE=7,  size_y=5),
        dict(sx=1016, sz=1000, SIZE=4,  size_y=4),
        dict(sx=1008, sz=1000, SIZE=7,  size_y=3),
        dict(sx=1017, sz=1009, SIZE=15, size_y=6),
        dict(sx=1000, sz=1009, SIZE=16, size_y=5),
    ]
    tile_to_block, dic = {}, {}
    for i, l in enumerate(LOCS):
        example, tile_to_block, dic = get_pattern_from_minecraft(**l, tile_to_block_dic=tile_to_block, dic=dic)
        shape = "_".join(map(str, example.shape))
        helpers.save_3d_example_to_file(InputPattern(example), f"examples/minecraft/{i}_{shape}.txt")

    with open("examples/minecraft/tile_dic.p", "wb+") as f:
        pickle.dump(
            {
                "tile_to_block": tile_to_block,
                "dic": dic,
            },
            f,
        )

if __name__ == '__main__':
    get_channel_and_client()
    read_all_patterns()
