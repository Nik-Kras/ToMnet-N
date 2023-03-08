"""
This module creates N games played by M agents
"""
import src.data.bot_ai as bot_ai
import src.data.read_map as read_map
import src.data.make_map as make_map

def main():
    
    ### Create N Maps, save them to `data/raw_maps`
    ### Iteratively call `bot_ai.py` to play each game
    ### Save each Game to `data/complete_games` 

    """
    create_maps_dataset(directory=`data/raw_maps`, ...)

    for i in ...:
        play_map(..., filename="game_{}".format(i))
    
    """
    pass

if __name__ == "__main__":
    main()