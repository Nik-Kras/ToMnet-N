"""
This module creates N games played by M agents
"""
import src.data.bot_ai as bot_ai
import src.data.read_map as read_map
import src.data.make_map as make_map


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

    """
    graph = Graph()
    map = load_map(...)
    graph.init_nodes(map)

    open_set = 
    """

    pass

def read_game(filename: str):
    """ loads 
    """
    pass


def create_games_dataset(path: str):
    """
    This function generates dataset of played games by agents

    :param path: Path to folder with maps to play
    :type path: int
    """
    for i in range(num_maps):
        generate_map(filename="map_{:04d}".format(i),
                     shape=shape,
                     num_goals=num_goals)
        if i % (num_maps/10) == 0:
            print("Progress {}%".format(int(100*i/num_maps)))

if __name__ == "__main__":

    create_maps_dataset(10)
    