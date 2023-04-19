from src.game_generation.utils.path_finder import get_trajectory
from src.game_generation.utils.map_processing import get_player_position, get_goals_coordinates, load_map, replace_element_on_map
from src.game_generation.utils.data_structures import GameData, Goals, Pos, MapElements
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Dict, List
import json

def load_agent_preferences(agent_type: str):
    """ Returns a dictionary with agent's goal values """
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'game_generation', 'utils'))
    with open(f"{output_dir}/agent_types.json") as json_file:
        agent_pref = json.load(json_file)
    return agent_pref[agent_type]

def get_agent_types():
    """ Returns a list of all agent's names / types """
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'game_generation', 'utils'))
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/agent_types.json") as json_file:
        agent_pref = json.load(json_file)
    return list(agent_pref.keys())

def get_trajectories(map: pd.DataFrame) -> Dict[str, List[Pos]]:
    """ Returns trajectories to each goal on the map """
    player_position = get_player_position(map)
    goals_coordinates = get_goals_coordinates(map)
    trajectories = {}

    for goal_name, goal_pos in goals_coordinates.items():
        
        # Replace all other goals with walls
        other_goals = goals_coordinates.copy()
        other_goals.pop(goal_name) 
        map_edited = map.copy()
        for other_goal_pos in other_goals.values():
            map_edited = replace_element_on_map(map_edited, other_goal_pos, MapElements.Wall)

        # Set start ad end coordinates
        start = player_position
        end = goal_pos

        # Get trajectory
        trajectories[goal_name] = get_trajectory(map_edited, start, end)

        # DEBUG
        # print("Start: {}, End: {}, Trajectory: {}".format(start, end, trajectories[goal_name]))
        # plt.axis('off')
        # plt.title("Map for {}".format(goal_name))
        # plt.imshow(map_edited.to_numpy(dtype=float))
        # plt.show()

    return trajectories

def select_trajectory(trajectories: dict, agent_type: str) -> Dict[str, int]:
    """ Applies Agent's preferences to select cost-effective goal consumption """
    goal_values = load_agent_preferences(agent_type)
    goal_costs = {key:len(value) for (key,value) in trajectories.items() if len(value) > 0} # len() = 0 -> No trajectory to goal

    cost_ef = {list(goal_costs.keys())[0]: -float("inf")}
    for goal, cost in goal_costs.items():
        efficience = goal_values[goal] - cost
        if list(cost_ef.values())[0] < efficience:
            cost_ef = {goal: efficience}

    output = {
        "goal": list(cost_ef.keys())[0],
        "score": list(cost_ef.values())[0],
        "trajectory": trajectories[goal]
    }

    # DEBUG
    # print("Values: {}".format(goal_values))
    # print("Costs: {}".format(goal_costs))
    # print("Best Goal: {}".format(cost_ef))

    return output

if __name__ == "__main__":
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'maps'))
    map = load_map(f"{output_dir}/map_0007.csv")
    trajectories = get_trajectories(map)
    select_trajectory(trajectories, "agent_1")
    select_trajectory(trajectories, "agent_2")
