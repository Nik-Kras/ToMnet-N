import numpy as np
import matplotlib.pyplot as plt

class TilesClass:
    """
    tiles - stores all possible states of cells representing patterns of walls
            for Grid World
    titleProbability - stores probability for each state to appear. It is a way to
                        adjust how many walls will be placed and which wall
                        structure would be prefered
    """

    def __init__(self):

        """
        List of all available tiles where Tile_0 ... Tile_6 are defined tiles
        and Tile_10 is a tile of undefined cells (which are not collapsed)
        """
        self.ListOfTiles = ["Tile_0", "Tile_1", "Tile_2", "Tile_3", "Tile_4",
                            "Tile_5", "Tile_6", "Tile_10"]

        self.tiles = {
            "Tile_0": np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]),

            "Tile_1": np.array([[0, 1, 0],
                                [0, 1, 1],
                                [0, 1, 0]]),

            "Tile_2": np.array([[0, 0, 0],
                                [1, 1, 1],
                                [0, 1, 0]]),

            "Tile_3": np.array([[0, 1, 0],
                                [1, 1, 0],
                                [0, 1, 0]]),

            "Tile_4": np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 0, 0]]),

            "Tile_5": np.array([[0, 0, 0],
                                [1, 1, 1],
                                [0, 0, 0]]),

            "Tile_6": np.array([[0, 1, 0],
                                [0, 1, 0],
                                [0, 1, 0]]),

            "Tile_10": np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]]),
        }

        self.tileProbability = [0] * 8
        self.tileProbability[0] = 26
        self.tileProbability[1] = 9
        self.tileProbability[2] = 9
        self.tileProbability[3] = 9
        self.tileProbability[4] = 9
        self.tileProbability[5] = 20
        self.tileProbability[6] = 20
        self.tileProbability[7] = 0

        """
        The dictionary that stores options for neighbours of each Tile.
        When a Tiles is assigned to a certain state, all neighbour options must
        be updated according to the rules stored here. Therefore, entropy is
        propagated among the cells.
        """
        self.Connection = {
            "wall": {
                "UP": ["Tile_0", "Tile_4", "Tile_5"],
                "RIGHT": ["Tile_0", "Tile_1", "Tile_6"],
                "DOWN": ["Tile_0", "Tile_2", "Tile_5"],
                "LEFT": ["Tile_0", "Tile_3", "Tile_6"]
            },
            "path": {
                "UP": ["Tile_1", "Tile_2", "Tile_3", "Tile_6"],
                "RIGHT": ["Tile_2", "Tile_3", "Tile_4", "Tile_5"],
                "DOWN": ["Tile_1", "Tile_3", "Tile_4", "Tile_6"],
                "LEFT": ["Tile_1", "Tile_2", "Tile_4", "Tile_5"]
            }
        }

        self.connectionRules = {
            "Tile_0": {
                "UP": self.Connection["wall"]["UP"],
                "RIGHT": self.Connection["wall"]["RIGHT"],
                "DOWN": self.Connection["wall"]["DOWN"],
                "LEFT": self.Connection["wall"]["LEFT"]
            },
            "Tile_1": {
                "UP": self.Connection["path"]["UP"],
                "RIGHT": self.Connection["path"]["RIGHT"],
                "DOWN": self.Connection["path"]["DOWN"],
                "LEFT": self.Connection["wall"]["LEFT"]
            },
            "Tile_2": {
                "UP": self.Connection["wall"]["UP"],
                "RIGHT": self.Connection["path"]["RIGHT"],
                "DOWN": self.Connection["path"]["DOWN"],
                "LEFT": self.Connection["path"]["LEFT"]
            },
            "Tile_3": {
                "UP": self.Connection["path"]["UP"],
                "RIGHT": self.Connection["wall"]["RIGHT"],
                "DOWN": self.Connection["path"]["DOWN"],
                "LEFT": self.Connection["path"]["LEFT"]
            },
            "Tile_4": {
                "UP": self.Connection["path"]["UP"],
                "RIGHT": self.Connection["path"]["RIGHT"],
                "DOWN": self.Connection["wall"]["DOWN"],
                "LEFT": self.Connection["path"]["LEFT"]
            },
            "Tile_5": {
                "UP": self.Connection["wall"]["UP"],
                "RIGHT": self.Connection["path"]["RIGHT"],
                "DOWN": self.Connection["wall"]["DOWN"],
                "LEFT": self.Connection["path"]["LEFT"]
            },
            "Tile_6": {
                "UP": self.Connection["path"]["UP"],
                "RIGHT": self.Connection["wall"]["RIGHT"],
                "DOWN": self.Connection["path"]["DOWN"],
                "LEFT": self.Connection["wall"]["LEFT"]
            }
        }

    """
    Draws all Tiles and shows probability for each of them
    """

    def DrawTiles(self):

        counter = 1
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Tiles", fontsize=16)

        for _, tile in self.tiles.items():
            ax = fig.add_subplot(3, 4, counter)
            ax.set_title("Tile_" + str(counter - 1))
            # ax.set_title("Prbability: " + str(self.tileProbability[counter-1]) + "%")
            plt.imshow(tile)
            counter = counter + 1
        plt.show()

    def getTile(self, index):
        if index in self.ListOfTiles:
            return self.tiles[index]
        else:
            print("Error! Wrong index was given. Expexted index: Tile_x where \
      x is 0, 1, ..., 6")
