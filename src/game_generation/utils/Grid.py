from Cell import Cell
from src.game_generation.utils.data_structures import Pos, MapElements
import pandas as pd
import matplotlib.pyplot as plt
        
class Grid:

    def __init__(self, shape):
        self.shape = shape
        self.Cells = [[Cell() for _ in range(shape)] for _ in range(shape)]

    def lowest_entropy_cell(self) -> Pos:
        """ Returns coordinates of not-collapsed cell with lowest entropy """
        coor = Pos(0,0)
        entropy = 10
        for x in range(self.shape):
            for y in range(self.shape):
                if not self.Cells[x][y].is_collapsed() and self.Cells[x][y].get_entropy() < entropy:
                    entropy = self.Cells[x][y].get_entropy()
                    coor = Pos(x,y)
        return coor

    def collapse_cell(self, coordinates: Pos):
        """ Assignes a pattern to cell by coordinates """
        self.Cells[coordinates.x][coordinates.y].collapse()

    def propagate_entropy(self, coordinates: Pos):
        """ Changes pattern oprions for all neighbours of given cell by coordinates """
        collapsed_cell = self.Cells[coordinates.x][coordinates.y]
        directions = [Pos(1,0), Pos(0,1), Pos(-1,0), Pos(0,-1)]
        for dir in directions:
            neighbour_coordinates = coordinates + dir
            if self.cell_is_inside_grid(neighbour_coordinates) and self.cell_is_not_collapsed(neighbour_coordinates):
                """
                I select each neighbour to the collapsed cell and change its entropy
                However, I inverse the direction as for this neighbour MY collapsed cell
                Will be a neighbour. It has opposite point of view which must be directed
                """
                x = neighbour_coordinates.x
                y = neighbour_coordinates.y
                print("\n---\n\nCollapsed Coordinates: ", str(coordinates))
                print("Propagated Coordinates: ", str(neighbour_coordinates))
                self.Cells[x][y].decrease_entropy(neighbour=collapsed_cell, 
                                                  direction=str(dir.inverse()))

    def cell_is_inside_grid(self, coordinates: Pos) -> bool:
        """ Checks if coordinates refer to valid cell """
        return 0 <= coordinates.x < self.shape and 0 <= coordinates.y < self.shape
    
    def cell_is_not_collapsed(self, coordinates: Pos) -> bool:
        """ Checks if Cell by coordinates is not collapsed yet """
        return not self.Cells[coordinates.x][coordinates.y].is_collapsed()

    def finished_generation(self):
        """ Checks if the map has finished generation process. True = finished """
        flat_cell_list = sum(self.Cells, [])
        for cell in flat_cell_list:
            if not cell.is_collapsed():
                return False
        return True
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """ Exports the Map from List[List[Cell]] to DataFrame of each pixel """
        map = pd.DataFrame(index=range(3*self.shape), columns=range(3*self.shape))
        for x in range(self.shape):
            for y in range(self.shape):

                # Each Cell
                for xx in range(3):
                    for yy in range(3):
                        map.loc[3*y + yy][3*x + xx] = self.Cells[x][y].get_pattern()[yy][xx]
        
        return map

if __name__ == "__main__":
    grid = Grid(shape = 9)
    grid.export_to_dataframe()      # Print empty DataFrame

    while not grid.finished_generation():
        coor = grid.lowest_entropy_cell()
        grid.collapse_cell(coor)
        grid.propagate_entropy(coor)

    map = grid.export_to_dataframe()
    plt.axis('off')
    plt.imshow(map.to_numpy(dtype=float))
    plt.show()