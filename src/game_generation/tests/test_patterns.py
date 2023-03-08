from src.game_generation.utils.Pattern import Patterns

def test_init():
    pattern = Patterns()
    assert len(pattern.patterns) == 0

def test_init_complete():
    pattern = Patterns()
    pattern.initialize_patterns()
    assert len(pattern.patterns) == 7

def test_remove():
    pattern = Patterns()
    pattern.initialize_patterns()
    assert len(pattern.patterns) == 7
    assert pattern.get_pattern("Pattern_4") is not None

    pattern.remove_pattern("Pattern_4")
    assert len(pattern.patterns) == 6
    assert pattern.get_pattern("Pattern_4") is None

def test_clear():
    pattern = Patterns()
    pattern.initialize_patterns()
    pattern.clear()
    assert len(pattern.patterns) == 0

