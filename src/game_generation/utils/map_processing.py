from src.game_generation.utils.data_structures import GameData, Goals, Pos, MapElements
import pandas as pd
import numpy as np
import os
from typing import Union, Dict

def load_map(filename: str) -> pd.DataFrame:
    """ Loads map into DataFrame """
    return pd.read_csv(filename, header=None)

def get_pos_of_element(map: pd.DataFrame, element: MapElements) -> Union[Pos, None]:
    """ Returns position of any element on the map or None if it was not found """
    if not map.isin([element.value]).any(axis=None):
        return None
    indexes = np.where(map.to_numpy()  == element.value)
    return Pos(indexes[1][0], indexes[0][0])

def get_player_position(map: pd.DataFrame) -> Pos:
    """ Returns position of the player on the map """
    player_position = get_pos_of_element(map, MapElements.Player)
    if not isinstance(player_position, Pos):
        raise ElementNotFound(MapElements.Player)
    return player_position

def get_goals_coordinates(map: pd.DataFrame) -> Dict[str,Pos]:
    """ Returns dictionary with Goals' names and their coordinates """
    goals = {}
    for goal in Goals:
        goal_pos = get_pos_of_element(map, goal)
        if isinstance(goal_pos, Pos):
            goals[goal.name] = goal_pos
    return goals
    
def replace_element_on_map(map: pd.DataFrame, pos: Pos, element: MapElements) -> pd.DataFrame:
    """ Replaces any element on the map with given element """
    return_map = map.copy()
    return_map.loc[pos.y][pos.x] = element.value
    return return_map

class ElementNotFound(Exception):
    """Exception raised when get_pos_of_element() is unable to find a required element

    Attributes:
        element -- MapElement
    """

    def __init__(self, element: MapElements):
        self.element = element
        self.message = "Element {} was not found!".format(element)
        super().__init__(self.message)


if __name__ == "__main__":

    # map = load_map("data/maps/map_0000.csv")
    # goal = get_pos_of_element(map, MapElements.Goal_A)

    # print(goal)

    map = pd.DataFrame([[0,1], [2,3]])
    print(map)
    map_2 = replace_element_on_map(map, Pos(0,0), MapElements.Goal_D)

    print(map)
    print(map_2)