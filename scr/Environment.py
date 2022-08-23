import numpy as np

"""
To use that environment do next:

1. Create an object based on the class and set the desired parameters of the game 
>>> import Environment
>>> game = Environment.GridWorld(tot_row = 30, tot_col = 30)
2. Create your own map of walls. 
- It must be a matrix of the same size as a game (30x30)
- The values of matrix have next meaning: 
    0  walkable path, 
    -1 wall
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
"""

class GridWorld:

    def __init__(self, tot_row, tot_col, goal_rewards=None, step_cost=-0.01):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col

        # Originally agent was started as random, I changed to be deterministic ( [0.5, 0.5] -> [1, 0] )
        #self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        self.transition_matrix = np.eye(self.action_space_size)

        # NOTE: state_matrix also holds player's position as number 10 in the matrix.
        # However, that should be removed in the future, but is left for rendering simplicity
        self.state_matrix = np.zeros((tot_row, tot_col))                          # Environmental Map of walls and goals
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]  # Indexes of Player position

        # Set the reward for each goal A, B, C, D.
        # It could differ for each agent,
        # So, at the beginning of the game it sets for an agent individually
        if goal_rewards is None:
            goal_rewards = [1, 2, 3, 4]
        self.goal_rewards = goal_rewards

        # Set step cost in the environment
        # It could differ from experiment to experiment,
        # So, should be set at the beginning of the game
        self.step_cost = step_cost

    """
        Clears all the map, preparing for a new one
    """
    def clear(self):
        self.state_matrix = np.zeros((self.world_row, self.world_col))
        self.position = [np.random.randint(self.world_row), np.random.randint(self.world_col)]
        self.transition_matrix = np.eye(self.action_space_size)

    def setTransitionMatrix(self, transition_matrix):
        """
        The transition matrix here is intended as a matrix which has a line
        for each action and the element of the row are the probabilities to
        executes each action when a command is given. For example:
        [[0.55, 0.25, 0.10, 0.10]
         [0.25, 0.25, 0.25, 0.25]
         [0.30, 0.20, 0.40, 0.10]
         [0.10, 0.20, 0.10, 0.60]]
        This matrix defines the transition rules for all the 4 possible actions.
        The first row corresponds to the probabilities of executing each one of
        the 4 actions when the policy orders to the robot to go UP. In this case
        the transition model says that with a probability of 0.55 the robot will
        go UP, with a probaiblity of 0.25 RIGHT, 0.10 DOWN and 0.10 LEFT.
        """
        if transition_matrix.shape != self.transition_matrix.shape:
            raise ValueError('The shape of the two matrices must be the same.')
        self.transition_matrix = transition_matrix

    def setStateMatrix(self, state_matrix):
        """Set the obstacles, player and goals in the world.
        The input to the function is a matrix with the
        same size of the world
        -1 for states which are not walkable.
        +1 for terminal states [+1, +2, +3, +4] - for 4 different goals
         0 for all the walkable states (non-terminal)
        The following matrix represents the 4x3 world
        used in the series "dissecting reinforcement learning"
        [[+3,  -1,   0,   +1]
         [0,   -1,   0,   +4]
         [0,    0,   0,   +2]]
        """
        if state_matrix.shape != self.state_matrix.shape:
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix

    def setPosition(self):
        """ Set the position of a player and 4 Goals randomly
            But only on a walkable cells.
            ! Before using this method make sure you generated walls and put them
              like game.setStateMatrix(state_matrix)
        """

        # Next objects must be placed on the path: Player, Goal 1, Goal 2, Goal 3, Goal 4
        objectsToPlace = [10, 1, 2, 3, 4]
        for obj in objectsToPlace:
            randomRow = np.random.randint(self.world_row)
            randomCol = np.random.randint(self.world_col)
            # Ensure that the obj is placed on the path
            # The coordinates will be changed until it finds a clear cell
            while self.state_matrix[randomRow][randomCol] != 0:
                randomRow = np.random.randint(self.world_row)
                randomCol = np.random.randint(self.world_col)
                print(self.state_matrix[randomRow][randomCol])
            self.state_matrix[randomRow, randomCol] = obj    # Record obj position on the map
            if obj == 10:
                self.position = [randomRow, randomCol]

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
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):

                # Draw
                if self.position == [row, col]: row_string += u" \u25CB " # u" \u25CC "

                # Draw player, walls, paths and goals
                else:
                    match self.state_matrix[row, col]:
                        # Wall
                        case -1:
                            row_string += ' # '
                        # Path
                        case 0:
                            row_string += ' - '
                        # Goal 1
                        case 1:
                            row_string += ' A '
                        # Goal 2
                        case 2:
                            row_string += ' B '
                        # Goal 3
                        case 3:
                            row_string += ' C '
                        # Goal 4
                        case 4:
                            row_string += ' D '
                        # Player
                        case 10:
                            row_string += u" \u25CB "  # u" \u25CC "

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
    def step(self, action, action_type="optimal"):
        """ One step in the world.
        [observation, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done True if the state is terminal
        """
        if action >= self.action_space_size:
            raise ValueError('The action is not included in the action space.')

        # Based on the current action and the probability derived
        # from the transition model it chooses a new action to perform
        if action_type == "random":
            # Picking randomly an action in accordance to the probabilities
            # Stored in transition_matrix
            action = np.random.choice(4, 1, p=self.transition_matrix[int(action), :])
        elif action_type == "optimal":
            # Picking the action with highest probability
            # Stored in transition_matrix
            action = action
        else:
            raise ValueError("The action_type is wrong!")

        # Check the boarders and
        # Move the player
        # Actions: 0 1 2 3 <-> UP RIGHT DOWN LEFT
        if   action == 0 and self.position[0] > 0:              new_position = [self.position[0]-1, self.position[1]  ]
        elif action == 1 and self.position[1] < self.world_col: new_position = [self.position[0],   self.position[1]+1]
        elif action == 2 and self.position[0] < self.world_row: new_position = [self.position[0]+1, self.position[1]  ]
        elif action == 3 and self.position[1] > 0:              new_position = [self.position[0],   self.position[1]-1]
        else: raise ValueError("Player goes out of the borders or the action is not included in the action space")

        # Check if player has hit the wall on its move...
        hit_wall = self.state_matrix[new_position[0], new_position[1]] == -1

        # NOTE: Redundant check, however, reduces risk if bug appears
        # Check if the new position is a valid position
        # ! if you go to the wall - the move is ignored and the cost of move is calculated!
        if 0 <= new_position[0] < self.world_row:
            if 0 <= new_position[1] < self.world_col:
                if not hit_wall:
                    self.position = new_position

        # to deal with variable visibility
        reward = 0

        # Return an occasion when the wall is hit
        if hit_wall:
            # Not False in case if the game didn't terminate on the goal cell
            done = bool(self.state_matrix[self.position[0], self.position[1]])
            reward = self.step_cost
            return self.position, reward, done

        # Otherwise calculate the reward for according to a new cell
        match self.state_matrix[self.position[0], self.position[1]]:
            case 0: reward = self.step_cost        # Path
            case 1: reward = self.goal_rewards[0]  # Goal 1
            case 2: reward = self.goal_rewards[1]  # Goal 2
            case 3: reward = self.goal_rewards[2]  # Goal 3
            case 4: reward = self.goal_rewards[3]  # Goal 4

        #Done is True if the state is a terminal state
        done = bool(self.state_matrix[self.position[0], self.position[1]])

        # should output self.state_matrix, reward, done
        return self.position, reward, done

