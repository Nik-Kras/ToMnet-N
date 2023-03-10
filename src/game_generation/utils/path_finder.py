from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from src.game_generation.utils.data_structures import Pos
import pandas as pd
from typing import List

def get_trajectory(map: pd.DataFrame, start: Pos, end: Pos) -> List[Pos]:
    """ Returns a trajectory from start to end point on the map """
    grid = Grid(matrix=map.to_numpy())

    start = grid.node(start.x, start.y)
    end = grid.node(end.x, end.y)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, _ = finder.find_path(start, end, grid)

    return [Pos(p[0], p[1]) for p in path]

if __name__ == "__main__":
    map = pd.DataFrame(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 1, 0]
        ]
    )
    start = Pos(0,3)
    end = Pos(0,0)
    path = get_trajectory(map, start, end)
    print(path)
