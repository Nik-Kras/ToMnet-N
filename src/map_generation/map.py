"""
API to generate maps / dataset of maps
"""
import pandas as pd
import os
from src.map_generation.utils.map_generator import generate_wfc_walls, put_goals_and_player, save_map
from src.map_generation.utils.drawer import draw_map



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
    if filename: save_map(complete_map, filename)
    return complete_map
    
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
    