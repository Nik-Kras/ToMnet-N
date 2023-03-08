from src.game_generation.utils.Cell import Cell

def test_collapse():
    cell = Cell()
    assert cell.get_entropy() == 7
    assert cell.is_collapsed() == False
    cell.collapse()
    assert cell.get_entropy() == 0
    assert cell.is_collapsed() == True

def test_entropy_propagate():
    neighbour_cell = Cell()
    neighbour_cell.set_pattern(name="Pattern_1")

    cell = Cell()
    cell.decrease_entropy(neighbour_cell, direction="left")
    assert cell.get_entropy() == 4

    cell = Cell()
    cell.decrease_entropy(neighbour_cell, direction="right")
    assert cell.get_entropy() == 3

    cell = Cell()
    cell.decrease_entropy(neighbour_cell, direction="up")
    assert cell.get_entropy() == 4

    cell = Cell()
    cell.decrease_entropy(neighbour_cell, direction="down")
    assert cell.get_entropy() == 4