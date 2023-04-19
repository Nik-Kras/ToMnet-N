"""
Map Generator stores implementation of map generation
"""
import pandas as pd
import numpy as np
import random
import os
from src.map_generation.utils.Grid import Grid
from src.map_generation.utils.data_structures import MapElements, Pos

from src.utils.logger import create_logger

# from utils.logger import create_logger

logger = create_logger("map_generator", "map_generator")


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
    logger.debug("Creating Grid..")
    grid = Grid(shape)
    grid.export_to_dataframe()      # Print empty DataFrame

    while not grid.finished_generation():
        coor = grid.lowest_entropy_cell()
        grid.collapse_cell(coor)
        grid.propagate_entropy(coor)

    logger.debug("WFC is finished")
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
        logger.error("You gave numer of goals: {}".format(num_goals))
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

def save_map(map: pd.DataFrame, filename: str):
    """ Saves map to data/maps """
    # Construct the absolute path to the output file
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'maps'))
    os.makedirs(output_dir, exist_ok=True)
    
    logger.debug("os.path.dirname(__file__) - {}".format(os.path.dirname(__file__)))
    logger.debug("os.path.join(x , '..', '..', '..', 'data', 'maps') - {}".format(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'maps')))
    logger.debug("os.path.abspath(x) - {}".format(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'maps'))))

    output_path = os.path.join(output_dir, f"{filename}.csv")
    
    logger.debug("Saving as: {}".format(output_path))

    # Save the output file to the absolute path
    map.to_csv(output_path, header=False, index=False)