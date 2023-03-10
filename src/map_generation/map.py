"""
Function generate_map()

Creates a random map of given shape
Uses Wave Function Collapse to generate walls
"""
import pandas as pd
import numpy as np
import random
from src.map_generation.utils.Grid import Grid
from src.map_generation.utils.data_structures import MapElements, Pos
from src.map_generation.utils.drawer import draw_map

def generate_wfc_walls(shape: int = 9) -> pd.DataFrame:
    """
    This function applies Wafe Function Collapse
    To generate random walls in the map shaped as `shape`
    Give it desired coordinates and receive a binary map 
    Map value description shown in `MapElements`

    :param shape: Two-positional tuple describing width and height of the map
    :type shape: typle
    :return: Binary map where 1 is wall and 0 is path
    :rtype: pd.DataFrame
    """
    grid = Grid(shape)
    grid.export_to_dataframe()      # Print empty DataFrame

    while not grid.finished_generation():
        coor = grid.lowest_entropy_cell()
        grid.collapse_cell(coor)
        grid.propagate_entropy(coor)

    map = grid.export_to_dataframe()
    return map

def put_goals_and_player(map: pd.DataFrame, num_goals: int) -> pd.DataFrame:
    """
    Give it a binary map where 1 means wall and 0 path
    And it will put `num_goals` goals and a player somewhere
    On the path in the map.
    - Goals are represented as integers started with 2.
    i.e. 4 goals would be: [2, 3, 4, 5]
    - Player is represented as integer

    :param map: Binary map where 1 means wall and 0 path. Goals and player will be placed in the map
    :type map: pd.DataFame
    :param num_goals: Number of goals to be placed in the map. Allowed range: from 1 to 8.
    :type num_goals: int
    :return: Map with walls, goals and player 
    :rtype: pd.DataFrame
    """
    if num_goals < 0 or num_goals > 7:
        raise ValueError("Select number of goals in the range from 1 to 7")
    
    empty_cells = np.where(map == MapElements.Empty.value)
    x, y = empty_cells
    random_empty_cells = random.sample(range(1, x.shape[0]), num_goals+1)

    # Put Goals 
    for count, random_ind in enumerate(random_empty_cells[:-1]):
        empty_x = x[random_ind]
        empty_y = y[random_ind]
        map.loc[empty_x][empty_y] = MapElements(count+2).value

    # Put Players
    empty_x = x[random_empty_cells[-1]]
    empty_y = y[random_empty_cells[-1]]
    map.loc[empty_x][empty_y] = MapElements.Player.value

    return map

def generate_map(filename: str, shape: int = 12,  num_goals: int = 4) -> pd.DataFrame:
    """
    This function generates complete map with walls, goals and player
    It returns the map as pd.DataFrame and saves it by given path

    :param filename: Path including desired filename for the map to be stored
    :type filename: str
    :param shape: Two-positional tuple describing width and height of the map
    :type shape: typle
    :param num_goals: Number of goals to be placed in the map. Allowed range: from 1 to 8.
    :type num_goals: int
    :return: Map with walls, goals and player 
    :rtype: pd.DataFrame
    """
    wall_map = generate_wfc_walls(shape)
    complete_map = put_goals_and_player(wall_map, num_goals)
    # draw_map(wall_map)
    # draw_map(complete_map)
    complete_map.to_csv("data/maps/{}.csv".format(filename), header=False, index=False)

def create_maps_dataset(num_maps: int, shape: int = 12, num_goals: int = 4):
    """
    This function generates `num_maps` complete maps with walls, goals and player
    It saves all of them in the given `directory` path with unique names for all of them.
    Use it to create a dataset of maps. Apply it to agents to generate game dataset for ToMnet-N

    :param num_maps: Number of maps to be generated
    :type num_maps: int
    :param shape: Two-positional tuple describing width and height of the map
    :type shape: typle
    :param num_goals: Number of goals to be placed in the map. Allowed range: from 1 to 8.
    :type num_goals: int
    """
    for i in range(num_maps):
        generate_map(filename="map_{:04d}".format(i),
                     shape=shape,
                     num_goals=num_goals)
        if i % (num_maps/10) == 0:
            print("Progress {}%".format(int(100*i/num_maps)))

if __name__ == "__main__":

    create_maps_dataset(5, shape=3)
    