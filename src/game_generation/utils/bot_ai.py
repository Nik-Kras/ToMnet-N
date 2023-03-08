"""
Function PlayGame()

that represents AI bot playing Grid World game

You set an agent type
It returns full trajectory and consumed goal
"""
import pandas as pd

def play_map(map: pd.DataFrame, agent_id: int, filename: str):
    """
    This function gets a map to play and type of agent to apply.
    It saves the game by given `filename` and also
    It returns a game with player's trajectory and consumed goal

    :param map: Map with walls, goals and player 
    :type map: pd.DataFrame
    :param agent_id: ID of specific agent to apply. Agents are specified in `agent_types.json`
    :type agent_id: int
    :param filename: Path including desired filename for the game to be stored
    :type filename: str
    """

    pass