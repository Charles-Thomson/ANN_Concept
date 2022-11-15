"""

https://www.oreilly.com/radar/reinforcement-learning-for-complex-goals-using-tensorflow/
Environment used by the Q Learning process

class MazeEnv()

init()

Variables

npMap - numpyArray -> takes the given map and coverts to np array.
RootOfMapSize- int -> The square root of the size of the map
npMap - reshape -> mpMap reshaped to the size value of RootOfMapSize, giving a
                   'square' of nested lists
nrows, ncols - int -> Number of rows and cols in npMap
goal_state_coords - tuple(int,int) -> The coords of the 'goal' node in the map
                                      (Value = 3 )

goal_state - int -> The single value denoting the state of the goal node, i.e
                    where in the 'map' using
                    a single int value

agent_start_coords - tuple -> The starting coords of the agent
agent_start_state - int -> The single value denoting the state of the agent
                           node, i.e where in the 'map' using
                    a single int value

agent_current_state/agent_current_coords -> the current state and coords
                                            of the agent

observation_space - int -> The number of possible states of the agent,
                           each node is a possible state.
action_space - int -> The number of possible acitons for the agent

EPISODE_LENGTH - int -> The length of each possible episode
                        i.e max possible steps before reset of agent
goal_reached - bool -> flag to indicate if the 'goal' node has been reached
                       by the agent

check_if_goal_reached()
Check if the goal node has been reached by the agent

step()
Called by the NN , takes a given action
-> action is mapped from an int value to a new x,y (row,col) coords value
-> EPISODE_LENGTH decrimented
-> termination flag check
-> Calculate reward
                   -> calls calculate_reward()
                   -> calls check_if_goal_reached()
-> returns agent_current_state(int) , reward(int) , terminated(bool),
           info(Filler), goal_reached(bool)

action_mapping()
-> takes an 'action'(int) and mapps to the relervent movemet of the agent

- Does not function as normal x,y as it starts top left
        - to move from (0,0) to (0,1) is to add to the column
        - to move from (0,0) to (1,0) is to add to the row
        - Your moving row/col not adding to the value of it if that makes sence
        - if "up" - take from row
        - if "down" - add to row
        - if "left" - take from col
        - if "right" - add to col

calculate_reward()
Calculates the reward given to the agent for moving to a specific state
step_bounus -> given to encorage movment by giving deminishing reward per step
case x -> x refers to the state of each node,
          i.e 2 is an obstical, 3 is the goal node

termination_check()
Checks a number of cases in which the agent will be terminated
-> try used as the call for .any() on np arracy casued issues

render ()
Renders elements of the data i.e he map and nodes states key

reset()
Reset the env
-> returns all vars to starting values
-> returns agent to start location
"""
import RNN_Helper_functions as HF
from gym import Env
from gym.spaces import Discrete


# Define custom env - inherit from gym env
class MazeEnv(Env):
    def __init__(self, MAP) -> None:

        print(MAP, type(MAP))

        # turn the map into a np array
        self.npMAP = HF.to_npMAP(MAP)

        # Number of rows and cols
        nrow, ncol = self.nrow, self.ncol = self.npMAP.shape

        self.goal_state = HF.find_goal_state(self.npMAP, ncol)
        self.goal_state_coords = HF.to_coords(self.goal_state, ncol)

        self.agent_start_state = HF.find_starting_state(self.npMAP, ncol)
        self.agent_start_coords = HF.to_coords(self.agent_start_state, ncol)

        # Agent current state
        self.agent_current_state = self.agent_start_state
        self.agent_current_coords = self.agent_start_coords

        # Obs space being the whole board
        self.observation_space = Discrete(self.nrow * self.ncol)

        # Number of actions
        self.action_space = Discrete(9)

        # Number of steps the agent can take
        self.EPISODE_LENGTH = 50

        self.goal_reached = False

    def check_if_goal_reached(self) -> None:
        if self.agent_current_state == self.goal_state:
            self.goal_reached = True

    def step(self, action: int) -> tuple[int, float, bool, list, bool]:

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

        # Mapps the action to the change in the state and the state coords
        # Returns -> new state
        # updates -> state_coords

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
    # rework based on step taken as time, gamma^time *r <- also gives the utility
    def calculate_reward(self) -> float:
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
        return 0

    # Terminate if episod length = 0, agent killed, agent reached end goal
    def termination_check(self) -> bool:

        try:
            at_obstical_location = (self.npMAP[self.agent_current_coords] == 2).any()
        except Exception:
            return True

        TERMINATION_CONDITIONS = [
            (self.agent_current_coords == None),
            (self.agent_current_state < 0),
            (self.agent_current_state > 25),
            (self.EPISODE_LENGTH == 0),
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
