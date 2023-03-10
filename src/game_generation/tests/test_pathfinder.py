from src.game_generation.utils.path_finder import get_trajectory
from src.game_generation.utils.data_structures import MapElements, Pos
import pandas as pd

def test_get_trajectory():
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

    assert len(path) == 8
    assert isinstance(path[0], Pos)