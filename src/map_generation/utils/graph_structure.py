from dataclasses import dataclass

@dataclass
class Pos():
	x: int
	y: int

@dataclass
class NodeType():
	Empty:  int  = 0
	Wall:   int  = 1
	Goal_A: int  = 2
	Goal_B: int  = 3
	Goal_C: int  = 4
	Goal_D: int  = 5
	Goal_E: int  = 6
	Goal_F: int  = 7
	Goal_G: int  = 8
	Player: int  = 9
	Goals = {Goal_A, Goal_B, Goal_C, Goal_D, Goal_E, Goal_F, Goal_G}
	

class Node:
	def __init__(self, 
                pos: Pos, 
                type: NodeType):
		self.pos: Pos = pos             # Coordinates (X, Y) on the Map
		self.type: NodeType = type      # Type of Cell (Empty, Wall, Goal, Player)
		self.traj_steps: int = 0        # Number of steps for a player to reach this Cell  
		self.path_from = None
		
class Graph:
	
	def __init__(self, 
	      		 shape: Pos):
		self.cell_type = NodeType()
		self.shape: Pos = shape

		# 2D map made of List[List[Cell]]
		self.graph = [[Node(pos = Pos(x,y), type = self.cell_type.Empty) for x in range(self.shape.x)] for y in range(self.shape.y)]

	def inMapBoundaries(self, position: Pos):
		""" Checks if position (x,y) is inside the map """
		return 0 <= position.x < self.shape.x and 0 <= position.y < self.shape.y

	def getNeighbors(self, node: Node):
		""" Returns coordinates of neighbour empty nodes """
		directions = [Pos(1,0), Pos(0,1), Pos(-1,0), Pos(0,-1)]
		neighbour_coorditanes = []
		for dir in directions:
			neighbour = Pos(x = dir.x + node.pos.x, 
		   					y = dir.y + node.pos.y)
			if self.inMapBoundaries(neighbour):
				neighbour_coorditanes.append(neighbour)
		return neighbour_coorditanes


if __name__ == "__main__":
	graph = Graph(Pos(3,3))

	print(len(graph.graph))
	print(len(graph.graph[1]))

	node_1 = Node(Pos(0,0), NodeType.Empty)
	node_2 = Node(Pos(1,1), NodeType.Empty)
	node_3 = Node(Pos(2,2), NodeType.Empty)
	node_4 = Node(Pos(3,3), NodeType.Empty)

	print(graph.getNeighbors(node_1))
	print(graph.getNeighbors(node_2))
	print(graph.getNeighbors(node_3))
	print(graph.getNeighbors(node_4))