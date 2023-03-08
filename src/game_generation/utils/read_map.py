"""
Function extract_features()

Reads a game from `data/complete_games`
And returns game in suitable for ToMnet-N format

It could be either Pydantic BaseModel or Pandas DataFrame
That keeps Map, Trajectory, Consumed Goal and Agent Type
"""

def read_game(filename: str):
    """
    Loads given game found by the `filename`
    Into the **** datatype (Pydantic BaseModel? Dict? pd.DataFrame?)

    :param filename: Path to a game 
    :type filename: str
    :return: The game found by the `filename` in the form of ****
    """
    pass


def load_dataset(directory: str):
    """
    Loads all games from the `directory` 
    Into the **** datatype (Pydantic BaseModel? Dict? pd.DataFrame?)

    :param directory: Path to a folder to store generated dataset 
    :type directory: str
    :return: All games from the directory in the form of ****
    """

    ## Get list of finemanes inside the directory
    ## Iteratively call read_game and append game to some data structure
    ## Return this data structure
    pass