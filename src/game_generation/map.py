"""
Function generate_map()

Creates a random map of given shape
Uses Wave Function Collapse to generate walls
"""
import pandas as pd

def generate_wfc_walls(shape: tuple) -> pd.DataFrame:
    """
    This function applies Wafe Function Collapse
    To generate random walls in the map shaped as `shape`
    Give it desired coordinates and receive a binary map where
    1 is wall and 0 is path

    :param shape: Two-positional tuple describing width and height of the map
    :type shape: typle
    :return: Binary map where 1 is wall and 0 is path
    :rtype: pd.DataFrame
    """
    """
    grid = Grid()

    while not grid.finished_generation():
        cell_coordinates = grid.lowest_entropy_cell()
        grid.collapse_cell(cell_coordinates)
        grid.propagate_entropy(cell_coordinates)

    binary map = grid.export_to_dataframe()  # [[]] or pd.DataFrame
    return binary_map
    """
    ### Check that shape is dividable by 3 for both height and width
    pass

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

    ### Check that `num_goals` is in the range (1,8)
    pass

def generate_map(filename: str, shape: tuple, num_goals: int = 4) -> pd.DataFrame:
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

    ## call generate_wfc_walls
    ## call put_goals_and_player
    ## save the map
    pass

def create_maps_dataset(directory: str, num_maps: int, shape: tuple, num_goals: int = 4):
    """
    This function generates `num_maps` complete maps with walls, goals and player
    It saves all of them in the given `directory` path with unique names for all of them.
    Use it to create a dataset of maps. Apply it to agents to generate game dataset for ToMnet-N

    :param directory: Path to a folder to store generated dataset 
    :type directory: str
    :param num_maps: Number of maps to be generated
    :type num_maps: int
    :param shape: Two-positional tuple describing width and height of the map
    :type shape: typle
    :param num_goals: Number of goals to be placed in the map. Allowed range: from 1 to 8.
    :type num_goals: int
    """

    ## Iteratively call `generate_map` `num_maps` number of times
    pass
