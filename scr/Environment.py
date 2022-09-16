import numpy as np
import random
from MapGenerator.Grid import *

# https://towardsdatascience.com/ai-learns-to-fly-part-2-create-your-custom-rl-environment-and-train-an-agent-b56bbd334c76

"""
To use that environment do next:

1. Create an object based on the class and set the desired parameters of the game 
>>> import Environment
>>> game = Environment.GridWorld(tot_row = 30, tot_col = 30)
2. Create your own map of walls. 
- It must be a matrix of the same size as a game (30x30)
- The values of matrix have next meaning: 
    1  walkable path, 
    0  wall
- Don't put anything else
- You could use Map Generator provided separately
>>> from MapGenerator.Grid import *
>>> Generator = Grid(SIZE)
>>> state_matrix = Generator.GenerateMap() - 1
3. Set the map according to your desired walls configuration
>>> game.setStateMatrix(state_matrix)
4. Set player and goals position randomly
>>> game.setPosition()
5. To view the world use
>>> game.render()
6. To read the world use
>>> game.getWorldState()
7. To make an action by agent use
>>> game.step(action) 
- That will return you observation of the world, 
- Therefore, at that moment you don't need any other functions besides step() and render()
8. To create a new game clear the environment
>>> game.clear()
9. Then, repeat from step #2

PS: How to read Actions:
    0 - UP
    1 - RIGHT
    2 - DOWN
    3 - LEFT

PS: How to read a Map:
    0 - Wall
    1 - Path
    2 - Goal A
    3 - Goal B
    4 - Goal C
    5 - Goal D
    10 - Player
    
PS: New Map:

    1. Player
        1 - Player
        0 - other
    2. Walls
        1 - Walls
        0 - other
    3. Goals
        1.0-inf - Goals (all goals values normalized?)
        0 - other
         
"""

class GridWorld():

    def __init__(self, tot_row, tot_col, goal_rewards=None, step_cost=-0.001, max_moves_per_episode=90):
        super().__init__()
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col

        # Indexes for 3 map layers
        self.PlayerMap = 0
        self.WallMap = 1
        self.GoalMap = 2

        # Set the values to represent objects on the Map
        self.ObjSym = {
            "Wall": 0,
            "Path": 1,
            "Goal A": 2,
            "Goal B": 3,
            "Goal C": 4,
            "Goal D": 5,
            "Player": 10,
        }

        self.MapSym =[
            # self.PlayerMap
            {"Player": 1,
             "Other": 0},
            # self.WallMap
            {"Wall": 0,
             "Other": 1},
            # self.GoalMap
            {"Goal A": 1,
             "Goal B": 2,
             "Goal C": 3,
             "Goal D": 4,
             "Other": 0}
        ]

        self.GoalValue = {
            self.ObjSym["Goal A"]: 2,
            self.ObjSym["Goal B"]: 4,
            self.ObjSym["Goal C"]: 8,
            self.ObjSym["Goal D"]: 16,
        }

        # Value of the goal <-> name of the goal
        self.ValueGoal = {
            self.GoalValue[self.ObjSym["Goal A"]]: "Goal A",
            self.GoalValue[self.ObjSym["Goal B"]]: "Goal B",
            self.GoalValue[self.ObjSym["Goal C"]]: "Goal C",
            self.GoalValue[self.ObjSym["Goal D"]]: "Goal D",
        }

        # Originally agent was started as random, I changed to be deterministic ( [0.5, 0.5] -> [1, 0] )
        #self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        self.transition_matrix = np.eye(self.action_space_size)

        self.state_matrix = [np.zeros((1,self.world_row, self.world_col,1), dtype=np.float16),
                             np.zeros((1,self.world_row, self.world_col,1), dtype=np.float16),
                             np.zeros((1,self.world_row, self.world_col,1), dtype=np.float16)]             # Environmental Map of walls and goals
        self.position = [np.random.randint(self.world_row), np.random.randint(self.world_col)]  # Indexes of Player position

        # Set the reward for each goal A, B, C, D.
        # It could differ for each agent,
        # So, at the beginning of the game it sets for an agent individually
        if goal_rewards is None:
            goal_rewards = [2, 4, 8, 16]
        self.goal_rewards = goal_rewards

        self.cnt_goal_picked = 0

        # Set step cost in the environment
        # It could differ from experiment to experiment,
        # So, should be set at the beginning of the game
        self.step_cost = step_cost

        # Max number of moves after which player
        self.max_moves = max_moves_per_episode

        self.reward_punish = -20 # Previously was -1

    """
        ################################  METHODS USED FOR TENSORFORCE  #####################################
    """

    # Shows specification on states data
    def states(self):
        # dict(type='int', shape=(self.world_row,self.world_col,), num_values=11)
        return dict(
            Player = dict(type='float', shape=(1, self.world_row, self.world_col)),
            Walls = dict(type='float',  shape=(1, self.world_row,  self.world_col)),
            Goals = dict(type='float',  shape=(1, self.world_row,  self.world_col))
        )

    # Shows specification on actions
    def actions(self):
        return dict(type='int', num_values=4)

    # Used for limiting episode with number of actions
    # Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()  # or put your number like 90 moves!

    # Conditions to stop / terminate the game
    # def terminal(self):
    #     self.terminal = # condition to go out of the border
    #     self.episode_end = # condition to reach 90 moves ()
    #     return self.finished or self.episode_end

    def reset(self):
        """Return initial_time_step."""

        #print("The RESET was called")

        # Clear the Map
        empty_map = np.zeros((3, self.world_row, self.world_col))
        self.setStateMatrix(empty_map, set="all")

        # Create a new Map
        Generator = Grid(int(self.world_row / 3))  # How many 3x3 tiles should be put in the Map
        walls = Generator.GenerateMap()
        # print("Created Walls Map: ", walls)
        self.setStateMatrix(walls, set="walls")

        # Put player and goals on the map
        self.setPosition()

        # Clear step counter in the game
        self.step_count = 0

        dict_map = {"Player": np.expand_dims(self.state_matrix[self.PlayerMap], axis=0),
                    "Walls":  np.expand_dims(self.state_matrix[self.WallMap], axis=0),
                    "Goals":  np.expand_dims(self.state_matrix[self.GoalMap], axis=0)}

        # In the future, it should output Observed map (7x7), not "self.state_matrix"
        return dict_map

    def get_sight(self, sight):
        simple_map = np.ones((self.world_row, self.world_col), dtype=np.int16)  # 0-wall, 1-path. Start with all path, then add walls, then add goals

        # Put walls
        for row in range(self.world_row):
            for col in range(self.world_col):
                if self.state_matrix[self.WallMap, row, col] == self.MapSym[self.WallMap]["Wall"]: simple_map[row, col] = self.ObjSym["Wall"]

        # Put goals
        for row in range(self.world_row):
            for col in range(self.world_col):
                if   self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal A"]: simple_map[row, col] = self.ObjSym["Goal A"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal B"]: simple_map[row, col] = self.ObjSym["Goal B"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal C"]: simple_map[row, col] = self.ObjSym["Goal C"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal D"]: simple_map[row, col] = self.ObjSym["Goal D"]

        result = np.full((sight, sight), None)

        half_sight = int(sight * 0.5)
        for i in range(sight):
            for j in range(sight):
                x = i + self.position[0] - half_sight
                y = j + self.position[1] - half_sight

                if -1 < x < self.world_row and -1 < y < self.world_col:
                    result[i, j] = simple_map[x, y]

        return self.position, result

    def check_terminate(self, action):

        # By default, everything is okay
        # If one of "bad" circumstances happens - it changes to True
        terminate = False
        goal_picked = None

        # If player made more than max_moves (90) steps - terminate the game
        if self.step_count > self.max_moves: return [True, goal_picked]
        #elif self.step_count == 0: print("First Move!")
        self.step_count += 1

        # Check the boarders and
        # Move the player
        # Actions: 0 1 2 3 <-> UP RIGHT DOWN LEFT
        if   action == 0 and self.position[0] > 0:                  new_position = [self.position[0] - 1, self.position[1]]
        elif action == 1 and self.position[1] < self.world_col - 1: new_position = [self.position[0], self.position[1] + 1]
        elif action == 2 and self.position[0] < self.world_row - 1: new_position = [self.position[0] + 1, self.position[1]]
        elif action == 3 and self.position[1] > 0:                  new_position = [self.position[0], self.position[1] - 1]
        else:
            #print("Player goes out of the borders")
            return [True, goal_picked] # This could be simplified to one big boolean expression instead of many if-else

        # Check if player has hit the wall on its move
        hit_wall = self.state_matrix[self.WallMap, new_position[0], new_position[1]] == self.MapSym[self.WallMap]["Wall"]  #self.ObjSym["Wall"]
        if hit_wall: return [True, goal_picked]

        # Check if player picked a goal
        if   self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Other"]:
            goal_picked = False  # Path
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal A"]:
            goal_picked = self.MapSym[self.GoalMap]["Goal A"]
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal B"]:
            goal_picked = self.MapSym[self.GoalMap]["Goal B"]
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal C"]:
            goal_picked = self.MapSym[self.GoalMap]["Goal C"]
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal D"]:
            goal_picked = self.MapSym[self.GoalMap]["Goal D"]
            # terminate   = True # Picking two goals changes behaviour of terminate

        return [terminate, goal_picked]

    def check_reward(self, action, terminate, goal_picked):

        # By default, the reward is just a cost of step by ground
        # It will be changed to higher or lower rewards depending on new position
        reward = self.step_cost

        # If gone out of border or stepped on the wall
        if (terminate == True) and (goal_picked == False): return self.reward_punish

        # Acquiring new position
        new_position = [self.position[0], self.position[1]] # Initialize variable
        if   action == 0:  new_position = [self.position[0] - 1, self.position[1]]
        elif action == 1:  new_position = [self.position[0], self.position[1] + 1]
        elif action == 2 : new_position = [self.position[0] + 1, self.position[1]]
        elif action == 3 : new_position = [self.position[0], self.position[1] - 1]
        else: print("ERROR: The action is incorrect. Must be between 0 and 3, got: ", action)

        # Check receiving the goal in the next step and taking according reward
        if   self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Other"]:  reward = self.step_cost        # Path
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal A"]: reward = self.goal_rewards[0]  # Goal 1
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal B"]: reward = self.goal_rewards[1]  # Goal 2
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal C"]: reward = self.goal_rewards[2]  # Goal 3
        elif self.state_matrix[self.GoalMap, new_position[0], new_position[1]] == self.MapSym[self.GoalMap]["Goal D"]: reward = self.goal_rewards[3]  # Goal 4
        else: print("ERROR: Incorrect map value! Position: ", new_position[0], ", ", new_position[1])

        return reward

    def move(self, action, terminate, goal_picked):

        # If gone out of border or stepped on the wall
        if (terminate == True) and (goal_picked == False): return self.state_matrix

        # Acquiring new position
        new_position = [self.position[0], self.position[1]]  # Initialize variable
        if   action == 0: new_position = [self.position[0] - 1, self.position[1]]
        elif action == 1: new_position = [self.position[0], self.position[1] + 1]
        elif action == 2: new_position = [self.position[0] + 1, self.position[1]]
        elif action == 3: new_position = [self.position[0], self.position[1] - 1]
        else: print("ERROR: The action is incorrect. Must be between 0 and 3, got: ", action)

        # Clear the current place of the player (COULD BE CHANGED WITH NEW ELEMENT TO SHOW TRAJECTORY)
        self.state_matrix[self.PlayerMap, self.position[0], self.position[1]] = self.MapSym[self.PlayerMap]["Other"] # self.ObjSym["Path"]

        # Update the player's position!
        self.position = new_position
        self.state_matrix[self.PlayerMap, self.position[0], self.position[1]] = self.MapSym[self.PlayerMap]["Player"] # self.ObjSym["Player"]

        dict_map = {"Player": np.expand_dims(self.state_matrix[self.PlayerMap], axis=0),
                    "Walls": np.expand_dims(self.state_matrix[self.WallMap], axis=0),
                    "Goals": np.expand_dims(self.state_matrix[self.GoalMap], axis=0)}

        return dict_map

    def execute(self, actions):

        action = actions

        if  0 >= action >= self.action_space_size:
            raise ValueError('The action is not included in the action space.')

        terminate, goal_picked = self.check_terminate(action)
        reward = self.check_reward(action, terminate, goal_picked)
        observe = self.move(action, terminate, goal_picked)

        # terminate = False # There is a new condition to terminate a game - pick two goals!

        # When the goal is picked - shaffle other goals
        if goal_picked and self.cnt_goal_picked == 0:
            # Delete picked goal
            self.delete_picked_goal()

            # Shaffle rest goals
            self.shaffle_goals()

            self.cnt_goal_picked += 1

        elif goal_picked and self.cnt_goal_picked == 1:
            terminate = True

        # if terminate: print("Episode is finished. Moves played: ", self.step_count, "Goal picked? ", goal_picked)

        return observe, terminate, goal_picked, reward

    """
            ################################  METHODS USED FOR TENSORFORCE  #####################################
    """

    def delete_picked_goal(self):
        self.state_matrix[self.GoalMap, self.position[0], self.position[1]] = self.MapSym[self.GoalMap]["Other"]

    def shaffle_goals(self):

        # 1. Create dictionary of left goals and their coordinates
        leftgoals = dict()
        map = self.state_matrix[self.GoalMap]
        Goals = ["Goal A", "Goal B", "Goal C", "Goal D"]

        for row in range(self.world_row):
            for col in range(self.world_col):
                if map[row, col] != self.MapSym[self.GoalMap]["Other"]:
                    index = int(map[row, col] - 1)
                    goal_name = Goals[index]
                    leftgoals[goal_name] = dict(position=[row, col], value=self.goal_rewards[index])

        # 2. Shaffle coordinates
        print("Goals before shuffle: ", leftgoals)
        # Random shaffle:
        # l = list(leftgoals.items())
        # random.shuffle(l)
        # leftgoals = dict(l)

        # Change least and most valuable goals:
        least_goal = dict(name="0", position=[0, 0], value=1000)
        most_goal  = dict(name="0", position=[0, 0], value=0)
        for key, leftgoal in leftgoals.items():
            if least_goal["value"] > leftgoal["value"]:
                least_goal["name"] = key # self.ValueGoal[leftgoal["value"]]
                least_goal["position"] = leftgoal["position"]
                least_goal["value"] = leftgoal["value"]
            if most_goal["value"] < leftgoal["value"]:
                most_goal["name"] = key # self.ValueGoal[leftgoal["value"]]
                most_goal["position"] = leftgoal["position"]
                most_goal["value"] = leftgoal["value"]

        print("The most valuable left goal: ", most_goal)
        print("The least valuable left goal: ", least_goal)

        new_leftgoals = dict()
        for key, leftgoal in leftgoals.items():
            if key == most_goal["name"]:
                leftgoal["position"] = least_goal["position"]
            elif key == least_goal["name"]:
                leftgoal["position"] = most_goal["position"]
            new_leftgoals[key] = leftgoal

        print("Goals after shuffle: ", new_leftgoals)

        # 3. Put new goals on the map
        map_new = np.zeros((self.world_row, self.world_col))
        for key, new_leftgoal in new_leftgoals.items():
            row = new_leftgoal["position"][0]
            col = new_leftgoal["position"][1]
            map_new[row, col] = self.MapSym[self.GoalMap][key]


        self.setStateMatrix(map_new, set="goals")

    """
        Clears all the map, preparing for a new one
    """
    def clear(self):
        self.reset()

    def setStateMatrix(self, state_matrix, set="all"):
        """Set the obstacles, player and goals in the world.
        The input to the function is a matrix with the
        same size of the world
        """
        if set=="all":
            # if state_matrix.shape != self.state_matrix.shape:
            #     raise ValueError('The shape of the matrix does not match with the shape of the world.')
            self.state_matrix = state_matrix
        elif set=="player":
            if state_matrix.shape != self.state_matrix[self.PlayerMap].shape:
                raise ValueError('The shape of the matrix does not match with the shape of the world.')
            self.state_matrix[self.PlayerMap] = state_matrix
        elif set == "walls":
            if state_matrix.shape != self.state_matrix[self.WallMap].shape:
                raise ValueError('The shape of the matrix does not match with the shape of the world.')
            self.state_matrix[self.WallMap] = state_matrix
        elif set=="goals":
            if state_matrix.shape != self.state_matrix[self.GoalMap].shape:
                raise ValueError('The shape of the matrix does not match with the shape of the world.')
            self.state_matrix[self.GoalMap] = state_matrix
        else:
            raise ValueError('The \'set\' parameter is wrong. Try: all, walls, player, goals')


    def setPosition(self):
        """ Set the position of a player and 4 Goals randomly
            But only on a walkable cells.
            !!! Before using this method make sure you generated walls and put them
                like game.setStateMatrix(state_matrix)
        """

        # 1. Put player on the map
        player_map = np.zeros((self.world_row, self.world_col), dtype=np.float16)
        cnt_of_tries = 0
        randomRow = 0
        randomCol = 0

        # Try random coordinates on the path, not walls
        while True:
            randomRow = np.random.randint(self.world_row)
            randomCol = np.random.randint(self.world_col)
            no_walls = self.state_matrix[self.WallMap, randomRow, randomCol] != self.MapSym[self.WallMap]["Wall"]
            # no_goals = elf.state_matrix[self.GoalMap, randomRow, randomCol] == self.MapSym[self.GoalMap]["Other"] # if no_walls and no_goals
            if no_walls:
                break

            # To prevent unsolvable maps (i.e. all walls)
            if cnt_of_tries > 10:
                self.reset()    # BUG: I should go out of loops, finish the function and then call reset(), as it will use setPosition automatically
                cnt_of_tries = 0
            else:
                cnt_of_tries += 1

        # Set the players position and record it on the map
        self.position = [randomRow, randomCol]  # Redundant
        player_map[randomRow, randomCol] = self.MapSym[self.PlayerMap]["Player"]
        self.setStateMatrix(player_map, set="player")

        # 2. Put Goals on the map
        goal_map = np.zeros((self.world_row, self.world_col), dtype=np.float16)
        Goals = self.MapSym[self.GoalMap].copy()
        Goals.popitem() # Remove "Others", so only Goal A - D are in the dictionary

        # EXAMPLE: key = "Goal A", value = 2
        for key, value in Goals.items():
            cnt_of_tries = 0
            while True:
                randomRow = np.random.randint(self.world_row)
                randomCol = np.random.randint(self.world_col)
                no_walls = self.state_matrix[self.WallMap, randomRow, randomCol] != self.MapSym[self.WallMap]["Wall"]
                no_goals = goal_map[randomRow, randomCol] == self.MapSym[self.GoalMap]["Other"]
                if no_walls and no_goals:
                    break

                # To prevent unsolvable maps (i.e. all walls)
                if cnt_of_tries > 10:
                    self.reset()    # BUG: I should go out of loops, finish the function and then call reset(), as it will use setPosition automatically
                    cnt_of_tries = 0
                else:
                    cnt_of_tries += 1

            goal_map[randomRow, randomCol] = self.MapSym[self.GoalMap][key]

        # print("Created Goal Map: ", goal_map)
        self.setStateMatrix(goal_map, set="goals")

        """
        # Next objects must be placed on the path: Player, Goal 1, Goal 2, Goal 3, Goal 4
        objectsToPlace = [self.ObjSym["Player"], self.ObjSym["Goal A"], self.ObjSym["Goal B"],
                          self.ObjSym["Goal C"], self.ObjSym["Goal D"]]
        for obj in objectsToPlace:
            randomRow = np.random.randint(self.world_row)
            randomCol = np.random.randint(self.world_col)
            # Ensure that the obj is placed on the path
            # The coordinates will be changed until it finds a clear cell
            cnt_of_tries = 0
            while self.state_matrix[randomRow][randomCol] != self.ObjSym["Path"]:
                randomRow = np.random.randint(self.world_row)
                randomCol = np.random.randint(self.world_col)

                # If after 10 attempts the object was not placed - rebuild the map!
                if cnt_of_tries > 10:
                    self.reset()

                cnt_of_tries += 1
                # print(self.state_matrix[randomRow][randomCol])
            self.state_matrix[randomRow, randomCol] = obj    # Record obj position on the map
            # Save the player's position in the separate variable (could be reduced)
            if obj == self.ObjSym["Player"]:
                self.position = [randomRow, randomCol]
                
        """

    def getWorldState(self):
        return self.state_matrix

    def getPlayerPosition(self):
        return self.position

    def render(self):
        """ Print the current world in the terminal.
        O           represents the player's position
        -           represents empty states.
        #           represents obstacles
        A, B, C, D  represent goals
        """
        # FUTURE DEVELOPMENT: I can create graph in first two cycles!

        # 1. Create 1xROWxCOL map from 3xROWxCOL
        simple_map = np.ones((self.world_row, self.world_col), dtype=np.int16)  # 0-wall, 1-path. Start with all path, then add walls, then add goals

        # Put walls
        for row in range(self.world_row):
            for col in range(self.world_col):
                if self.state_matrix[self.WallMap, row, col] == self.MapSym[self.WallMap]["Wall"]: simple_map[row, col] = self.ObjSym["Wall"]

        # Put goals
        for row in range(self.world_row):
            for col in range(self.world_col):
                if   self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal A"]: simple_map[row, col] = self.ObjSym["Goal A"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal B"]: simple_map[row, col] = self.ObjSym["Goal B"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal C"]: simple_map[row, col] = self.ObjSym["Goal C"]
                elif self.state_matrix[self.GoalMap, row, col] == self.MapSym[self.GoalMap]["Goal D"]: simple_map[row, col] = self.ObjSym["Goal D"]


        # 2. Draw a Map
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):

                # Draw player
                if self.position == [row, col]: row_string += u" \u25CB " # u" \u25CC "

                # Draw walls, paths and goals
                else:
                    if   simple_map[row, col] == self.ObjSym["Wall"]:   row_string += ' # '  # Wall
                    elif simple_map[row, col] == self.ObjSym["Path"]:   row_string += ' - '  # Path
                    elif simple_map[row, col] == self.ObjSym["Goal A"]: row_string += ' A '  # Goal 1
                    elif simple_map[row, col] == self.ObjSym["Goal B"]: row_string += ' B '  # Goal 2
                    elif simple_map[row, col] == self.ObjSym["Goal C"]: row_string += ' C '  # Goal 3
                    elif simple_map[row, col] == self.ObjSym["Goal D"]: row_string += ' D '  # Goal 4
                    else: print("ERROR: Incorrect map value! Position: " ,row, ", ", col)

            row_string += '\n'
            graph += row_string
        print(graph)

    """
        According to Open AI principles applied to Gym package - 
        Step function should:
            Do: make an action that agent wants in the environment
            Output:
                - New observation of the world (the whole world or limited section)
                - Collected reward after applying an agent's step
                - Status if the game is terminated or not (if the goal is reached - the game is done!)
    """