"""
The env for the next step of the RL project

- Maintain the map
- Pass back data of what can been seen in each direction for the agent

Model free RL only uses the current state values to make a preddiction
Model based RL is trying to make an action based
               on the future tate of the model

"""
import Maze_ENV_Helper_Functions as HF
import numpy as np
from gym import Env
from gym.spaces import Discrete
import CustomLogging as CL

actions_by_step_logging = CL.GenerateLogger(
    name=__name__, Log_File="LoggingStepByAction.log"
)

ENV_MAP = [
    [2, 1, 1, 1, 2],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [2, 1, 1, 1, 3],
]


class MazeEnv(Env):
    def __init__(self, agent_start_state: int, ENV_MAP: list = ENV_MAP):
        self.map = np.array(ENV_MAP)

        nrow, ncol = self.nrow, self.ncol = self.map.shape
        self.agent_start_state = agent_start_state

        self.EPISODE_LENGTH = 10

        self.observation_space = Discrete(nrow * ncol)
        self.action_space = Discrete(9)

    def reset(self):
        agent_state = self.agent_start_state
        self.EPISODE_LENGTH = 5
        return agent_state

    def action_mapping(self, action: int, agent_state: int) -> tuple[int, bool]:
        hrow, hcol = HF.to_coords(agent_state, self.ncol)

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

        if self.check_bondries(hrow, hcol) is False:
            # No move due to out of bounds -> Terminate
            print("Terminating in bounds")
            return (agent_state, True)

        return (HF.to_state((hrow, hcol), self.ncol), False)

    def check_bondries(self, hrow, hcol) -> bool:
        """
        Returns True if the move is in bounds else False
        """

        BOUDRY_CONDITIONS = [
            (0 <= hrow),
            (hrow < self.ncol),
            (0 <= hcol),
            (hcol < self.nrow),
        ]

        if all(BOUDRY_CONDITIONS):
            return True

        return False

    def calculate_reward(self, agent_state: int):
        value_at_state = HF.get_loaction_value(
            self.map, HF.to_coords(agent_state, self.ncol)
        )
        reward = 0

        match value_at_state:
            case 1:  # Open Tile
                reward = 0.1 + self.EPISODE_LENGTH / 100

            case 2:  # Obstical
                reward = 0

            case 3:  # goal
                reward = 100

        return reward

    def step(self, agent_state: int, action: int) -> tuple[int, float, list, bool]:
        i: list = []
        self.EPISODE_LENGTH -= 1
        ns, t_a = self.action_mapping(action, agent_state)
        t_b: bool = self.termination_check(ns)
        r: int = self.calculate_reward(agent_state)

        if t_a or t_b:
            t = True
        else:
            t = False

        actions_by_step_logging.debug(
            f"Agent State: {agent_state} - Action: {action} - New State: {ns} - Termination: {t} "
        )

        return ns, r, i, t

    def termination_check(self, ns: int) -> bool:

        agent_coords = HF.to_coords(ns, self.ncol)
        value_at_state = HF.get_loaction_value(self.map, agent_coords)

        TERMINATION_CONDITIONS = [
            (value_at_state == 3),
            (value_at_state == 2),
            (self.EPISODE_LENGTH < 0),
        ]

        if any(TERMINATION_CONDITIONS):
            print("Termination in main check")
            return True

        return False

    # // ------------------------------------------------------------- //

    # Call to HF get_location_value
    def get_location_value_call(self, coords: tuple) -> int:
        map = self.map
        value = HF.get_loaction_value(map, coords)
        return value

    def to_coords_call(self, state: int) -> tuple:
        ncol = self.ncol
        coords = HF.to_coords(state, ncol)
        return coords

    def to_state_call(self, coords: tuple) -> int:
        ncol = self.ncol
        state = HF.to_state(coords, ncol)
        return state


MazeEnv(agent_start_state=12)
