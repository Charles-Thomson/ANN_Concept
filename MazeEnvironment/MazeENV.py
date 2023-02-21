import numpy as np
from gym import Env

# from Decorator import __init__reset
import copy
import HyperPerameters

import Logging.CustomLogging as CL

actions_by_step_logging = CL.GenerateLogger(
    name=__name__, Log_File="LoggingStepByAction.log"
)


"""
    MazeEnv(Env) 
        param: Env : class : Inherited from the Gym module

    __init__(episode_length, env_map, agent_start_state)
        decorator: __init__reset.resettable : Copys the inital dict of the object, resets to the save dict on call 

        param: episode_length : int : Number of actions per episode by an agent
        param: env_map : list[list[int]] : Maze used in the current instance, taken as perdefined
        param : agent_start_state : int : Starting state of the agent in the env, taken as perdefined

        var : env_map : np.array : np.array of the maze used in the current instance
        var : self.agent_state : int : Current state of the agent
        var : self.episode_length : int : Number of actions per episode by an agent in this instance
        var : self.goal_reached_flag : bool : Indicates if the goal state has been reached during th current episode

    reset(self)
        Reset the object varibles to those saved by the __init_reset.resettable decorator

    get_agent_state(self) -> int
        Get the current state of the agent

        rtn : int : Returns the curent sate of the agent

    action_mapping(self, action, agent_state) -> tuple[int,int]
        Converts the given action (int) to a move in the env by the agent

        param: action : int : The given action
        param : agent_state : int : The current state of the agent

        var : hrow : int : The temp value of the row the agent occupies
        var : hcol : int : The temp value of the column the agent occupies

        rtn : tuple(hrow,hcol) : Returns the row & column variables after the action has been applied

    calculate_reward(agent_state) -> float
        Calculate the reward given based on the agents last action

        param : agent_state : int : The current state of the agent

        var : value_at_state : int : The value in the environment relating to the agents current state

        rtn : float : Float value based on the agents new state

    step(agent_state, action) -> tuple [int, float, list, bool , bool]
        Process the next step by the agent and determine the reward given, if the action results in termination
        or the reaching of the goal state

        param : agent_state : int : The current state of the agent
        param: action : int : The given action by the agent

        var : new_hrow : int : The new row the agent occupies
        var : new_hcol : int : The new column the agent occupies
        var : t : bool : If the agent is to be terminated
        var : ns : int : New state of the agent follwoing the action
        var : r : float : Reward for taking the action
        var : g : bool : If the goal state has been reached
        var : i : list : Requiered by the Gym module - not used

        rtn : tuple [int, float, list, bool , bool] : returns the related information from taking the given action 

    termination_check(new_hrow, new_hcol) -> bool
        Check if the new state, following the action, is valid i relation to the boundries of the env 
        and the termination/goal states. Will also Terminat if the episode_duration has elapsed.

        var : new_hrow : int : The new row the agent occupies
        var : new_hcol : int : The new column the agent occupies

        rtn: bool : Returns True if the agent is to be terminated

    // Helper Functions

    get_location_value(map, coords) -> int
        returns the value at a given location(state) in the env

        param : map : np.array : The map of the env
        param : coords : tuple[int, int] : The location to be checked

        rtn: int : Returns the value at the given coords

    to_state(coords, ncol) -> int 
        converts given coordinates in the env to the corisponding state value

        param : coords : tuple[int, int] : The coordinates to be converted to state value
        param : ncol : int : The number of columns in the env map

        rtn : int : returns the state at the given coordiates in the env

    to_coords(state, ncol) -> tuple[int,int]
        Converts a given state to its corisponding coordinates in the env

        param : sate : int : The state in the env to be converted
        param : ncol : int : The numebr of columns in the env map

        rtn : tuple(int, int) : Returns the coordinates of the given state in the env
"""


def resettable(func):
    def __init_and_copy__(self, *args, **kwargs):
        func(self, *args)
        self.__origional_dict__ = copy.copy(self.__dict__)

        def reset(o=self):
            o.__dict__ = o.__origional_dict__

        self.reset = reset

    return __init_and_copy__


class MazeEnv(Env):
    @resettable
    def __init__(
        self,
        episode_length: int = HyperPerameters.episode_length,
        env_map: list = HyperPerameters.ENV_MAP,
        agent_start_state: int = HyperPerameters.agent_start_state,
    ):
        self.env_map = np.array(env_map)
        self.nrow, self.ncol = self.env_map.shape
        print(self.nrow, self.ncol)
        self.agent_state = agent_start_state
        self.episode_length = episode_length
        self.states_visited = []

    def reset(self):
        actions_by_step_logging.debug(f"/n ")
        self.agent_state = HyperPerameters.agent_start_state
        self.episode_length = HyperPerameters.episode_length
        self.states_visited = []
        # self.__init__()

    def get_agent_state(self) -> int:
        return self.agent_state

    def get_env_shape(self):
        return self.nrow, self.ncol

    def action_mapping(self, action: int, agent_state: int) -> tuple[int, bool]:
        hrow, hcol = self.to_coords(agent_state, self.ncol)

        match action:
            case 0:  # Up + Left
                hrow -= 1
                hcol -= 1

            case 1:  # Up
                hrow -= 1

            case 2:  # Up + Right
                hrow -= 1
                hcol += 1

            case 3:  # left
                hcol -= 1

            case 4:  # No Move
                pass

            case 5:  # Right
                hcol += 1

            case 6:  # Down + Left
                hrow += 1
                hcol -= 1

            case 7:  # Down
                hrow += 1

            case 8:  # Down + Right
                hcol += 1
                hrow += 1

        return (hrow, hcol)

    def calculate_reward(self, agent_state: int):
        value_at_state = self.get_loaction_value(
            self.env_map, self.to_coords(agent_state, self.ncol)
        )

        match value_at_state:
            case 1:  # Open Tile
                return 0.1 + self.episode_length / 100

            case 2:  # Obstical
                return 0

            case 3:  # goal
                return 3

    def step(self, agent_state: int, action: int) -> tuple[int, float, list, bool]:

        self.i: list = []

        self.episode_length -= 1
        new_hrow, new_hcol = self.action_mapping(action, agent_state)

        t: bool = self.termination_check(new_hrow, new_hcol)

        ns: int = self.to_state((new_hrow, new_hcol), self.ncol)
        # ns: int = agent_state if t else self.to_state((new_hrow, new_hcol), self.ncol)

        r: int = (
            0 if self.states_used(agent_state) else self.calculate_reward(agent_state)
        )

        actions_by_step_logging.debug(
            f"Agent State: {agent_state} - Action: {action} - New State: {ns} - Termination: {t} Reward: {r} row,col {new_hrow, new_hcol}"
        )

        return ns, r, self.i, t

    def states_used(self, state: int) -> bool:
        if state in self.states_visited:
            return True

        self.states_visited.append(state)
        return False

    def termination_check(self, new_hrow: int, new_hcol: int) -> bool:

        TERMINATION_CONDITIONS = [
            (0 > new_hrow),
            (new_hrow >= self.ncol),
            (0 > new_hcol),
            (new_hcol >= self.nrow),
            (self.episode_length == 0),
        ]

        if any(TERMINATION_CONDITIONS):
            self.i.append(f"Termination: Bounds/Episode   - {new_hrow} / {new_hcol}")
            return True

        if self.get_loaction_value(self.env_map, (new_hrow, new_hcol)) == 2:
            self.i.append(f"Termination: Obstical - {new_hrow} / {new_hcol}")
            return True

        return False

    # // ------------------------------------------------------------- // HELPER FUNCTIONS

    def get_loaction_value(self, map: np.array, coords: tuple) -> int:
        x, y = coords
        value = map[x][y]
        return value

    def to_state(self, coords: tuple, ncol: int) -> int:
        state = (coords[0] * ncol) + coords[1]
        return state

    def to_coords(self, state: int, ncol: int) -> tuple:
        x = int(state / ncol)
        y = int(state % ncol)
        return (x, y)
