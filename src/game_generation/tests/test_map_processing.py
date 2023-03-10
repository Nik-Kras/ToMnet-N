from src.game_generation.utils.map_processing import get_pos_of_element, load_map, replace_element_on_map, get_goals_coordinates
from src.game_generation.utils.data_structures import MapElements, Pos
import pandas as pd

def test_get_pos_of_element_player():
    map = load_map("src/game_generation/tests/map_0000.csv")
    player = get_pos_of_element(map, MapElements.Player)

    assert player.x == 8
    assert player.y == 16

def test_get_pos_of_element_goal_true():
    map = load_map("src/game_generation/tests/map_0000.csv")
    goal = get_pos_of_element(map, MapElements.Goal_A)

    assert goal.x == 6
    assert goal.y == 34

def test_get_pos_of_element_goal_false():
    map = load_map("src/game_generation/tests/map_0000.csv")
    goal = get_pos_of_element(map, MapElements.Goal_G)

    assert goal is None

def test_replace_element_on_map():
    map = pd.DataFrame([[0,1], [2,3]])
    map_2 = replace_element_on_map(map, Pos(0,0), MapElements.Goal_D)

    assert map.loc[0][0] == 0
    assert map_2.loc[0][0] == MapElements.Goal_D.value

def test_get_goals_coordinates():
    map = load_map("src/game_generation/tests/map_0000.csv")
    goals = get_goals_coordinates(map)

    assert MapElements.Goal_A.name in goals
    assert MapElements.Goal_B.name in goals
    assert MapElements.Goal_C.name in goals
    assert MapElements.Goal_D.name in goals
    assert MapElements.Goal_E.name not in goals
    assert MapElements.Goal_F.name not in goals
    assert MapElements.Goal_G.name not in goals

    assert goals[MapElements.Goal_A.name].x == 6
    assert goals[MapElements.Goal_A.name].y == 34