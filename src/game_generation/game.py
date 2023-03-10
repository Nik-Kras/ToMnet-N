"""
This module creates N games played by M agents
"""
from src.game_generation.utils.map_processing import load_map
from src.game_generation.utils.agent_player import get_trajectories, get_agent_types, select_trajectory
from src.game_generation.utils.data_structures import GameData
import pandas as pd
import os

def create_games_dataset(path_maps: str):
    map_filenames = [f for f in os.listdir(path_maps) if os.path.isfile(os.path.join(path_maps, f))]
    print(map_filenames)
    
    for map_name in map_filenames:

        ### Generate one game per Agent?
        ### Or generate all maps for all Agents?
        ### As for now, I will use each map for all agents, but it could be changed in time
        map = load_map(os.path.join(path_maps, map_name))
        trajectories = get_trajectories(map)

        ### Change this to random.choice() to have one map per agent
        agent_types = get_agent_types()
        for agent in agent_types:
            selected_trajectory = select_trajectory(trajectories, agent)
            game = GameData(agent_type=agent, 
                            goal_consumed=list(selected_trajectory.keys())[0],
                            trajectory=list(selected_trajectory.values())[0],
                            map=map)
            game.save_game()

if __name__ == "__main__":

    # create_games_dataset("data/maps")

    x = pd.DataFrame()
    