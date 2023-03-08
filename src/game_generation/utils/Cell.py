from __future__ import annotations
from src.game_generation.utils.Pattern import Patterns, Pattern
from typing import Literal

class Cell:

    def __init__(self):         
        self._options = Patterns()
        self._options.initialize_patterns()
        self._state = self._options.get_pattern("Pattern_Default")

    ### Use it only for Testing!!!
    def set_pattern(self, name):
        """ Sets given pattern to the cell """
        self._state = self._options.get_pattern(name)

    def get_pattern(self):
        """ Returns assigned pattern aka state """
        return self._state

    def get_entropy(self) -> int:
        """ Returns number of possible patterns fro the cell aka entropy """
        return len(self._options.patterns)

    def is_collapsed(self) -> bool:
        """ Checks if the Cell is collapsed """
        return len(self._options.patterns) == 0

    def collapse(self):
        """ Sets one pattern from the list of possible patterns """
        self._state = self._options.get_random_pattern()
        self._options.clear()

    def decrease_entropy(self, neighbour: Cell, direction: Literal["left", "right", "up", "down"]):
        """ Updates list of possible patters due to collapse of neighbour cell """
        print("Direction Propagated->Collapsed: ", direction)
        print("Neighbour: \n", str(neighbour.get_pattern())) 
        print("Patterns #1: \n", str(self._options))
        self._options.patterns = [pattern for pattern in self._options.patterns if self.pattern_valid(pattern, neighbour, direction)]
        print("Patterns #2: \n", str(self._options))

    def pattern_valid(self, pattern: Pattern, neighbour: Cell, direction: Literal["left", "right", "up", "down"]) -> bool:
        """ Checks if the patter can be applied to the cell if there is a specific neighbour on certain direction """
        n_pattern = neighbour.get_pattern()
        if direction == "left":
            return all([pattern.state[i][0] == n_pattern[i][2] for i in range(3)])
        elif direction == "right":
            return all([pattern.state[i][2] == n_pattern[i][0] for i in range(3)])
        elif direction == "up":
            return all([pattern.state[0][i] == n_pattern[2][i] for i in range(3)])
        elif direction == "down":
            return all([pattern.state[2][i] == n_pattern[0][i] for i in range(3)])
        