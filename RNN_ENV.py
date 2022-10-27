import numpy as np
from gym import Env
from gym.spaces import Discrete


# Define custom env - inherit from gym env
class MazeEnv(Env):
    def __init__(self, MAP) -> None:

        # turn the map into a np array
        self.npMAP = np.array(MAP)

        # Number of rows and cols
        self.nrow, self.ncol = self.npMAP.shape

        # Goal State
        goal_state_coords = np.where(self.npMAP == 3)
        self.goal_state_coords = (int(goal_state_coords[0]), int(goal_state_coords[1]))

        self.goal_state = (
            self.goal_state_coords[0] * self.ncol
        ) + self.goal_state_coords[
            1
        ]  #  ncol * x  + y

        # Agent starting state
        agent_start_coords = np.where(self.npMAP == 0)

        self.agent_start_coords = (
            int(agent_start_coords[0]),
            int(agent_start_coords[1]),
        )  # Array type to int

        self.agent_start_state = (
            self.agent_start_coords[0] * self.ncol + self.agent_start_coords[1]
        )

        # Agent current state
        self.agent_current_state = self.agent_start_state
        self.agent_current_coords = self.agent_start_coords

        # Obs space being the whole board
        self.observation_space = Discrete(self.nrow * self.ncol)

        # Number of actions i.e  move; top left, up top right ect
        self.action_space = Discrete(9)

        # Number of steps the agent can take
        self.EPISODE_LENGTH = 50

        self.goal_reached = False

    def check_if_goal_reached(self) -> None:
        if self.agent_current_state == self.goal_state:
            self.goal_reached = True

    def step(self, action: int) -> tuple[int, int, bool, dict, bool]:

        # Update the self.state based on the given action
        self.action_mapping(action)

        # Reduce the number of remaining steps
        self.EPISODE_LENGTH -= 1

        # Check for termination
        terminated = self.termination_check()

        # Calculate reward
        if terminated:
            reward = 0
        else:
            reward = self.calculate_reward()
            self.check_if_goal_reached()

        # place holder - requiered by gym
        info = []

        return self.agent_current_state, reward, terminated, info, self.goal_reached

    def action_mapping(self, action: int) -> None:
        # Currently hard coded to a 5x5 map - the inverse addition is given as te map starts top left as 0,0 -
        # to go down adds to y to go up takes from y

        # Mapps the action to the change in the state and the state coords
        # Returns -> new state
        # updates -> state_coords

        """
        Hurt my brain with this one
        - Does not function as normal x,y as it starts top left
        - to move from (0,0) to (0,1) is to add to the column
        - to move from (0,0) to (1,0) is to add to the row
        - Your moving row/col not adding to the value of it if that makes sence
        - if it hs "up" - take from row
        - if it hs "down" - add to row
        - if it hs "left" - take from col
        - if it hs "right" - add to col

        """

        row, col = self.agent_current_coords
        state = self.agent_current_state

        # print(f"Current State: {state} - Current Coords: {row,col}")
        # print(f"Next action: {action}")

        match action:
            case 0:  # up + left
                self.agent_current_coords = (row - 1, col - 1)
                self.agent_current_state = state - 6

            case 1:  # up
                self.agent_current_coords = (row - 1, col)
                self.agent_current_state = state - 5

            case 2:  # up + right
                self.agent_current_coords = (row - 1, col + 1)
                self.agent_current_state = state - 4

            case 3:  # Left
                self.agent_current_coords = (row, col - 1)
                self.agent_current_state = state - 1

            case 4:  # Stay
                pass

            case 5:  # Right
                self.agent_current_coords = (row, col + 1)
                self.agent_current_state = state + 1

            case 6:  # Down + Left
                self.agent_current_coords = (row + 1, col - 1)
                self.agent_current_state = state + 4

            case 7:  # Down
                self.agent_current_coords = (row + 1, col)
                self.agent_current_state = state + 5

            case 8:  # Down + Right
                self.agent_current_coords = (row + 1, col + 1)
                self.agent_current_state = state + 6

    # Reward given per step, per goal and for end goal
    def calculate_reward(self) -> int:
        current_environment_location = self.npMAP[self.agent_current_coords]

        step_bonus = (51 - self.EPISODE_LENGTH) / 100

        match current_environment_location:
            case 0:
                return step_bonus + 1
            case 1:
                return step_bonus + 1
            case 2:
                return 0
            case 3:
                return step_bonus + 1000

    # Terminate if episod length = 0, agent killed, agent reached end goal
    def termination_check(self) -> bool:

        try:
            at_obstical_location = (self.npMAP[self.agent_current_coords] == 2).any()
        except:
            return True

        TERMINATION_CONDITIONS = [
            (self.agent_current_coords == None),
            (self.agent_current_state < 0),
            (self.agent_current_state > 25),
            (self.EPISODE_LENGTH == 0),
            # (self.agent_current_coords == self.goal_node), goal reached condition causing issue in termination
            (self.agent_current_coords[0] < 0),
            (self.agent_current_coords[0] > 25),
            (self.agent_current_coords[1] < 0),
            (self.agent_current_coords[1] > 25),
            (at_obstical_location),
        ]

        if any(TERMINATION_CONDITIONS):
            return True

        return False

    # Visualize env
    def render(self):
        print(self.npMAP)
        print(
            """
        Key; 
        0 - Agent Start 
        1 - Open Tile
        2 - Wall / Obstruction
        3 - Finish / Goal
        4 - Agent State / Agent Location 
        """
        )

    # Rest env
    def reset(self) -> int:
        self.agent_current_state = self.agent_start_state
        self.agent_current_coords = self.agent_start_coords
        self.EPISODE_LENGTH = 100
        self.goal_reached = False

        return self.agent_current_state
