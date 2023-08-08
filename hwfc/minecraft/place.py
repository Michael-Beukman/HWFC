import pickle
from hwfc.minecraft.mc_utils import clear

from hwfc.utils.helpers import load_example_from_file

import hwfc.external.minecraft_pb2_grpc as minecraft_pb2_grpc
import grpc
import numpy as np
from hwfc.external.minecraft_pb2 import *
import hwfc.generators.hierarchical_wfc


channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)
def place_in_mc(world, LOC=None, correct_tiles=False, correct_unfilled=False):
    from hwfc.world import SuperPosition
    with open("examples/minecraft/tile_dic.p", "rb") as f:
        d = pickle.load(f)
        tile_to_block = d["tile_to_block"]
        dic = d["dic"]
    START_X = -50
    START_Z = 0
    if LOC is None:
        client.fillCube(FillCubeRequest(
            cube=Cube(
                max=Point(x=-25, y=24, z=25),
                min=Point(x=-50, y=4,  z=0)
            ),
            type=AIR
        ))
        client.fillCube(FillCubeRequest(
            cube=Cube(
                max=Point(x=-25, y=4, z=25),
                min=Point(x=-50, y=4,  z=0)
            ),
            type=MOSSY_COBBLESTONE
        ))
    else:
        START_X = LOC[0]
        START_Z = LOC[1]
    
    
    blocks_list = []
    is_world = isinstance(world, hwfc.world.World)
    if is_world:
        world.map = world.map[:, ::-1]
        map = world.map
        COORDS = np.argwhere(map != AIR)
    else: 
        map = world
        COORDS = np.argwhere(np.ones(map.tiles.shape, dtype=np.int32))
        map = map.tiles[:, ::-1]
        client.fillCube(FillCubeRequest(  # Clear a 20x10x20 working area
        cube=Cube(
            max=Point(x=START_X + map.shape[0] - 1, y=4, z=START_Z+ map.shape[2] - 1),
            min=Point(x=START_X, y=4,  z=START_Z)
        ),
        type=MOSSY_COBBLESTONE
    ))

    for i, j, k in (COORDS):
        T: SuperPosition = map[i, j, k]
        if is_world and not T.is_collapsed:
            mc_block = AIR
        else:
            block_to_place = tile_to_block[T.possible_tiles[0]] if is_world else tile_to_block[T]
            if correct_tiles:
                if correct_unfilled and block_to_place == BLACK_GLAZED_TERRACOTTA:
                    block_to_place = AIR
                elif block_to_place == RED_GLAZED_TERRACOTTA:
                    block_to_place = AIR
                elif block_to_place in [GOLD_BLOCK, IRON_BLOCK, REDSTONE_BLOCK]:
                    block_to_place = SANDSTONE
                elif block_to_place == LAPIS_BLOCK:
                    block_to_place = WOODEN_DOOR
            mc_block = block_to_place

                
        
        blocks_list.append(
        Block(position=Point(x=START_X + i, y=5 + j, z= START_Z+ k), type=int(mc_block), orientation=NORTH),
        )
        if len(blocks_list) >= 100:
            client.spawnBlocks(Blocks(blocks=blocks_list))
            blocks_list = []
    client.spawnBlocks(Blocks(blocks=blocks_list))

def init_mc():
    level_files = [
        "all_exps/main/minecraft/Hierarchical_WFC/a/final_map.p",
        "all_exps/main/minecraft/Hierarchical_WFC/b/final_map.p",
        "all_exps/main/minecraft/Hierarchical_WFC/c/final_map.p",
    ]

    alls = [
        ['examples/minecraft/1_7_5_7.txt', 'examples/minecraft/3_7_3_7.txt', 'examples/minecraft/2_4_4_4.txt'],
        ['examples/minecraft/5_16_5_16.txt', 'examples/minecraft/4_15_6_15.txt',],
        ['examples/minecraft/0_25_25_25.txt'],
        level_files
    ]
    
    # Place examples
    
    x_start = x_start_og = 1000
    z_start              = 1000
    
    clear(x_start, z_start, 200, 200)
    def load_world(p):
        with open(p, "rb") as f: 
            return pickle.load(f)
    for jj, row in enumerate(alls):
        max_z = 0
        for example in row:
            if 'map.p' not in example:
                is_pattern = True
                ex = load_example_from_file(example)
            else:
                is_pattern = False
                ex = load_world(example)

            x, y, z = ex.shape
            max_z = max(max_z, z)
            client.fillCube(FillCubeRequest(
                cube=Cube(
                    min=Point(x=x_start,           y=4,  z=z_start),
                    max=Point(x=x_start + x - 1,   y=4,  z=z_start + z - 1)
                ),
                type=MOSSY_COBBLESTONE
            ))
            client.fillCube(FillCubeRequest(
                cube=Cube(
                    min=Point(x=x_start,           y=5,  z=z_start),
                    max=Point(x=x_start + x - 1,   y=24,  z=z_start + z - 1)
                ),
                type=AIR
            ))
            print(example, f"dict(sx={x_start}, sz={z_start}, SIZE={x}, size_y={y})")
            place_in_mc(ex, LOC=(x_start, z_start), correct_tiles=not is_pattern)
            
            x_start += x + 1
        z_start += max_z + 2
        if jj == 2: z_start += 40
        x_start = x_start_og

if __name__ == '__main__':
    init_mc()