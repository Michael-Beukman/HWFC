import pickle
from typing import Dict, Tuple

import external.minecraft_pb2_grpc as minecraft_pb2_grpc
import grpc
import numpy as np
from external.minecraft_pb2 import *

from hwfc.utils.classes import Tile
from hwfc.utils.mytypes import SUPER_POS, TOT_TILES
from hwfc.world import World

channel = None
client = None


def get_channel_and_client():
    global channel, client
    channel = grpc.insecure_channel("localhost:5001")
    client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


def clear(sx, sz, size_x, size_z):
    get_channel_and_client()
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=sx,          y=4,  z=sz),
            max=Point(x=sx + size_x, y=24, z=sz + size_z),
        ),
        type=AIR
    ))

    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=sx,          y=3,  z=sz),
            max=Point(x=sx + size_x, y=3, z=sz + size_z),
        ),
        type=GRASS
    ))

def get_pattern_from_minecraft(
    sx: int = 0, sz: int = 0, SIZE: int = 25, size_y: int = None, tile_to_block_dic: Dict[Tile, str]={}, dic: Dict[str, int]={}
) -> Tuple[np.ndarray, Dict[Tile, str], Dict[str, int]]:
    """This reads a pattern from a location in Minecraft

    Args:
        sx (int, optional): The starting x location. Defaults to 0.
        sz (int, optional): The starting z location. Defaults to 0.
        SIZE (int, optional): The size of the cube to read. Defaults to 25.
        size_y (int, optional): If given, overrides the y (i.e. vertical/air) height. Defaults to None.
        tile_to_block_dic (Dict[Tile, str], optional): This is part of what is returned, keeps a mapping between minecraft tiles and Tile(). Defaults to {}.
        dic (Dict[str, int], optional): This is part of what is returned, keeps a mapping between minecraft tiles and Tile(). Defaults to {}.

    Returns:
        Tuple[np.ndarray, Dict[Tile, str], Dict[str, int]]: The map, and the two mapping dictionaries that need to be passed into the next call of this function
    """
    
    if size_y is None:
        size_y = SIZE
    blocks = client.readCube(
        Cube(min=Point(x=sx, y=4 + 1, z=sz), max=Point(x=sx + SIZE - 1, y=size_y + 4 + 1 - 1, z=sz + SIZE - 1))
    )

    arr = np.zeros((SIZE, size_y, SIZE), dtype=np.int32)
    K = len(tile_to_block_dic)

    for b in blocks.blocks:
        type = b.type
        if type == DIRT: type = GRASS
        
        if type == BLACK_GLAZED_TERRACOTTA:
            dic[type] = SUPER_POS.value

        if type not in dic:
            dic[type] = K
            K += 1
        num = dic[type]

        tile_to_block_dic[Tile(num, TOT_TILES)] = type

        x = b.position.x
        y = b.position.y
        z = b.position.z
        arr[x - sx, y - 4 - 1, z - sz] = num

    new_arr = []
    for t in arr.flatten():
        new_arr.append(Tile(t, TOT_TILES))

    new_arr = np.array(new_arr).reshape(arr.shape)
    return new_arr[:, ::-1], tile_to_block_dic, dic

def seed_minecraft_world(world: World) -> World:
    # Seed the ground level to have a stone boundary and one tile of air above the stone.
    with open("examples/minecraft/tile_dic.p", "rb") as f:
        d = pickle.load(f)
        tile_to_block = d["tile_to_block"]
        block_to_tile = {v: k for k, v in tile_to_block.items()}
    stone_tile_val = block_to_tile[STONE]
    air_tile_val   = block_to_tile[AIR]
    assert stone_tile_val.value == 1
    assert air_tile_val.value   == 2
    
    for i in range(world.shape[0]):
        tt = stone_tile_val
        world.map[i, -1, -1].set([tt], True)
        world.map[i, -1, 0] .set([tt], True)
        world.map[-1, -1, i].set([tt], True)
        world.map[0, -1, i] .set([tt], True)
        tt = air_tile_val
        world.map[i, -2, -1].set([tt], True)
        world.map[i, -2, 0] .set([tt], True)
        world.map[-1, -2, i].set([tt], True)
        world.map[0, -2, i] .set([tt], True)
    return world

