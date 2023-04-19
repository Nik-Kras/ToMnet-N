from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import json

@dataclass
class Pos():
    x: int
    y: int
        
    def __add__(self, other: Pos) -> Pos:
        """ Implements addition of coordinates """
        x = self.x + other.x
        y = self.y + other.y
        return Pos(x, y)
    
    def __str__(self) -> str:
        """ Says which direction the vector points """
        if self.x == 0 and self.y == 1:
            return "down"
        elif self.x == 0 and self.y == -1:
            return "up"
        elif self.x == 1 and self.y == 0:
            return "right"
        elif self.x == -1 and self.y == 0:
            return "left"
        else:
            return "x: {}, y: {}".format(self.x, self.y)
        
    def inverse(self) -> Pos:
        """ Returns inverse vector """
        return Pos(-self.x, -self.y)
    
class PosEncoder(json.JSONEncoder):
    """ Serializes Pos to print trajectories with json.dumps() """
    def default(self, o: Any) -> Any:
        if isinstance(o, Pos):
            return {'x': o.x, 'y': o.y}
        return super().default(o)

class MapElements(Enum):
    
    Wall   = 0
    Empty  = 1
    Goal_A = 2
    Goal_B = 3
    Goal_C = 4
    Goal_D = 5
    Goal_E = 6
    Goal_F = 7
    Goal_G = 8
    Player = 9
