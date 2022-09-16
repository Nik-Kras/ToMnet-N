from MapGenerator.Tile import *

class Cell:
    """
    options - shows which states (from 0 to 6) are available for this cell
    collapsed - shows if the cell was collapsed, which means the state was defined
    state - shows which state was assigned to the cell (from 0 to 6), where
            10 means the state was not assigned yet
    """

    def __init__(self):
        self.options = [
            "Tile_0",
            "Tile_1",
            "Tile_2",
            "Tile_3",
            "Tile_4",
            "Tile_5",
            "Tile_6"
        ]
        self.collapsed = False
        self.state = "Tile_10"
        self.entropy = len(self.options)

    def getState(self):
        return self.state

    def getEntropy(self):
        self.entropy = len(self.options)
        return self.entropy

    def isCollapsed(self):
        return self.collapsed

    def setState(self, newState='Tile_0', method="direct"):

        if self.isCollapsed():
            print("The cell is already collapsed!")
            return

        if method == "random":
            newState = np.random.choice(self.options)
            # print("All options: ", self.options)
            # print("Random selection: ", newState)

        temporalTileObj = TilesClass()

        if newState in temporalTileObj.ListOfTiles:
            self.state = newState
        elif type(newState) == int:
            if 0 <= newState <= 6:
                self.state = temporalTileObj.ListOfTiles[newState]
            else:
                print("Error. The state index is out of range (0,6)")
        else:
            print("Error. Wrong state was given. Neither Tile name, nor Tile index")

        self.options = []
        self.entropy = 0
        self.collapsed = True
