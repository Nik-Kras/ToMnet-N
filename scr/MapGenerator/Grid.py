from MapGenerator.Cell import *

class Grid:
    """

    """

    def __init__(self, size):
        self.size = size
        self.Tiles = TilesClass()
        self.Cells = np.ndarray(shape=(size, size), dtype=Cell)
        self.CollapsedCells = 0
        self.Map = np.zeros(shape=(3 * size, 3 * size))

        for i in range(size):
            for j in range(size):
                self.Cells[i][j] = Cell()

    def DrawBoard(self, includeEntropy=False, tiles="separate", title=""):

        if tiles == "separate":
            counter = 1
            fig = plt.figure(figsize=(8, 8))
            if isinstance(title, str):
                fig.suptitle("Tiles", fontsize=16)
            elif isinstance(title, int):
                fig.suptitle(str(title) + "%", fontsize=16)


            for rowCell in self.Cells:
                for cell in rowCell:
                    cellState = cell.getState()

                    ax = fig.add_subplot(self.size, self.size, counter)
                    # ax.set_title(cellState)

                    if includeEntropy:
                        plt.text(0.7, 0.7, str(cell.getEntropy()), fontsize=12, color='w')

                    plt.axis('off')
                    plt.imshow(self.Tiles.getTile(cellState))

                    counter = counter + 1
            # fig.tight_layout()
            plt.show()
        elif tiles == "unite":
            plt.axis('off')
            plt.imshow(self.Map)
            plt.show()
        else:
            print("Error. Wrong tiles value was given!")

    """
    The function returns a cell with a lowest entropy
    """

    def LowestEntropy(self):
        lowestEntropy = 7
        lowestEntropyIndex = [0, 0]
        # previousCellCollapse = self.Cells[0][0].isCollapsed()

        for i in range(self.size):
            for j in range(self.size):
                cell = self.Cells[i][j]
                if not cell.isCollapsed() and cell.getEntropy() < lowestEntropy:
                    lowestEntropy = cell.getEntropy()
                    lowestEntropyIndex = [i, j]

        # print("Cell with lowest entropy: ", lowestEntropyIndex)
        # print("Is the cell collapsed?",
        #      self.Cells[lowestEntropyIndex[0]][lowestEntropyIndex[1]].isCollapsed())
        return lowestEntropyIndex

    def UpdateCellOptions(self, cellIndex, availableOptions):
        row = cellIndex[0]
        column = cellIndex[1]
        # print("Available options: ", availableOptions)
        # print("My options: ", self.Cells[row][column].options)

        if self.Cells[row][column].isCollapsed():
            # print("This cell is already collapsed")
            return
        else:
            copyOptions = self.Cells[row][column].options.copy()
            for option in copyOptions:
                if option in availableOptions:
                    continue
                else:
                    # print("I deleted an option: ", option)
                    self.Cells[row][column].options.remove(option)

            # print("Cell [", row, "][", column, "]. My new options: ", self.Cells[row][column].options)

    """
    cellIndex = [i, j] - Index (row and collumn) of the cell which neighbours
                          shpuld be updated
    """

    def UpdateOptionsOfOthers(self, cellIndex):
        row = cellIndex[0]
        column = cellIndex[1]
        collapsedCell = self.Cells[row][column]
        cellState = collapsedCell.getState()

        # Update cell above
        if row > 0:
            availableOptions = self.Tiles.connectionRules[cellState]["UP"]
            self.UpdateCellOptions([row - 1, column], availableOptions)

        # Update cell below
        if row < self.size - 1:
            availableOptions = self.Tiles.connectionRules[cellState]["DOWN"]
            self.UpdateCellOptions([row + 1, column], availableOptions)

        # Update cell to the right
        if column < self.size - 1:
            availableOptions = self.Tiles.connectionRules[cellState]["RIGHT"]
            self.UpdateCellOptions([row, column + 1], availableOptions)

        # Update cell to the right
        if column > 0:
            availableOptions = self.Tiles.connectionRules[cellState]["LEFT"]
            self.UpdateCellOptions([row, column - 1], availableOptions)

    def CollapseCell(self, cellIndex):
        row = cellIndex[0]
        column = cellIndex[1]
        self.Cells[row][column].setState(method="random")

    """
    Collapse one cell with the lowest entropy and changes available options 
    of neighbours (makes update according to assigned state)
    """

    def Update(self):
        # Chose the cell with lowest entropy
        index = self.LowestEntropy()
        # Collapse the cell, select one state for it
        self.CollapseCell(index)
        # Propagate entropy to neighbours, change their available options
        self.UpdateOptionsOfOthers(index)
        self.CollapsedCells = self.CollapsedCells + 1

    def GenerateMap(self, drawStages=False):
        maxNumberCollapsedCells = int(self.size * self.size)
        percentTHreshold = 10
        while self.CollapsedCells < maxNumberCollapsedCells:
            self.Update()
            percent = 100 * self.CollapsedCells / maxNumberCollapsedCells
            if percent > percentTHreshold or percent == 100:
                # print(f"The map is generated by {percent:.1f}%")
                if drawStages:
                    self.DrawBoard(includeEntropy=True, title=percentTHreshold)
                percentTHreshold = percentTHreshold + 10

        # Fill 2D array to save the whole map
        for i in range(self.size):
            for j in range(self.size):
                cell = self.Cells[i][j]
                state = cell.state
                cell2D = self.Tiles.tiles[state]

                for ii in range(3):
                    for jj in range(3):
                        row = i * 3 + ii
                        col = j * 3 + jj
                        self.Map[row][col] = cell2D[ii][jj]
        return self.Map