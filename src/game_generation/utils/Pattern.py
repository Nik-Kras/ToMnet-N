from enum import Enum
import random

class Pattern:
    """
    Represents a pattern of a cell.

    A pattern is a 3x3 binary matrix representing a possible state.
    The weight attribute is used to set the probability of selecting this pattern.
    """

    def __init__(self, state, weight, name = "Pattern_N"):
        self.state = state
        self.weight = weight
        self.name = name

    def __str__(self):
        return "State: {}, weight: {}, name: {}".format(self.state, self.weight, self.name)

class Patterns:
    """
    Represents a collection of patterns.

    A collection of patterns can be used to randomly set the pattern of a cell based on probability weights.
    """

    def __init__(self):
        self.patterns = []

    def __str__(self):
        return "\n".join([str(pattern) for pattern in self.patterns])
    
    def initialize_patterns(self):
        """ Init possible patterns with given pre-set patterns """
        self.patterns.append(Pattern([[0, 1, 0], [0, 1, 1], [0, 1, 0]], 2, "Pattern_1"))
        self.patterns.append(Pattern([[0, 0, 0], [1, 1, 1], [0, 1, 0]], 2, "Pattern_2"))
        self.patterns.append(Pattern([[0, 1, 0], [1, 1, 0], [0, 1, 0]], 2, "Pattern_3"))
        self.patterns.append(Pattern([[0, 1, 0], [1, 1, 1], [0, 0, 0]], 2, "Pattern_4"))
        self.patterns.append(Pattern([[0, 0, 0], [1, 1, 1], [0, 0, 0]], 1, "Pattern_5"))
        self.patterns.append(Pattern([[0, 1, 0], [0, 1, 0], [0, 1, 0]], 1, "Pattern_6"))
        self.patterns.append(Pattern([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 0, "Pattern_Default"))

    def get_random_pattern(self):
        """ Returns a random pattern based on probability weights """
        return random.choices(population=[x.state for x in self.patterns], 
                              weights=(x.weight for x in self.patterns), 
                              cum_weights=None, 
                              k=1)[0]
    
    def remove_pattern(self, name):
        """ Removes a pattern from the list by name """
        for p in self.patterns:
            if p.name == name:
                self.patterns.remove(p)

    def get_pattern(self, name = "Pattern_Default"):
        """ Removes a pattern from the list by name """
        for p in self.patterns:
            if p.name == name:
                return p.state
        return None
    
    def clear(self):
        """ Clears options, resets an object """
        self.patterns = []

if __name__ == "__main__":
    p = Patterns()
    print("Patterns: \n{}".format(p))
    p.initialize_patterns()
    print("Patterns: \n{}".format(p))