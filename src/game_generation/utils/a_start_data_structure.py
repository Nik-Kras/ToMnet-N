from __future__ import annotations
from src.game_generation.utils.data_structures import Pos, MapElements
import pandas as pd
from enum import Enum
"""
A* theory

F = G + H
G - Shortest distance from Start to this Node
H - Estimated distance from this Node to End Node

"""	
def map_element_to_node_type(map_element: MapElements) -> NodeType:
	""" Returns corresponding Node Type to Map Element """
	if map_element == MapElements.Wall:
		return NodeType.Wall
	elif map_element == MapElements.Empty:
		return NodeType.Empty
	elif map_element == MapElements.Player:
		return NodeType.Start
	else:
		return NodeType.End

class NodeType(Enum):
	Start = 1
	Empty = 2
	Wall = 3
	End = 4

class Node:

	def __init__(self, pos: Pos):
		self.position = pos
		self.g_distance = float("inf")
		self.h_distance = float("inf")
		self.last_node = Pos(0,0)
		self.node_type = NodeType.Empty

	def init_node_type(self, node_type: NodeType):
		""" Sets node type """
		self.node_type = node_type

	def init_h_distance(self, end_node_position: Pos):
		""" Calculates Heuristic distance to given End Node """
		self.h_distance = self.position.distance_manhattan(end_node_position)

	def get_g_distance(self):
		""" Returns shortest distance from Start to this Node """
		return self.g_distance
	
	def get_f_distance(self):
		""" Returns F distance """
		return self.g_distance + self.h_distance
	
	def set_node(self, last_node: Pos, shortest_distance: int):
		""" Sets new shortest distance and updates last node """
		self.last_node = last_node
		self.g_distance = shortest_distance
		
class Graph:
	
	def __init__(self, shape: int):
		self.shape = shape
		self.Nodes = [[Node() for _ in range(shape)] for _ in range(shape)]

	def init_nodes(self, map: pd.DataFrame):
		""" Initialize Nodes with values according to given map """
		### map MUST HAVE ONLY ONE GOAL.
		### IF MAP HAS MORE THAN ONE GOAL -> 
		### ITERATIVELY REPLACE THEM WITH WALLS AND CALL THIS FUNCTION

		# Init Node Types
		goal_pos = None
		for i in range(self.shape):
			for j in range(self.shape):
				node_type = map_element_to_node_type(MapElements(map.loc[i][j]))
				self.Nodes[i][j].init_node_type(node_type)
				if node_type == NodeType.End:
					goal_pos = Pos(i,j)

		### TODO: It could be remvoed here for speed efficiency
		# Init Heuristic Distance
		for i in range(self.shape):
			for j in range(self.shape):
				self.Nodes[i][j].init_h_distance(goal_pos)

