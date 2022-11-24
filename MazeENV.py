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

ENV_MAP = [
    [2, 1, 2, 1, 3],
    [1, 1, 2, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1],
]


class MazeEnv(Env):
    def __init__(self, agent_start_state: int, ENV_MAP: list = ENV_MAP):
        self.map = np.array(ENV_MAP)

        nrow, ncol = self.nrow, self.ncol = self.map.shape

        self.agent_state = agent_start_state

        self.set_agent_start(agent_start_state)

        self.EPISODE_LENGTH = 50

        self.observation_space = Discrete(nrow * ncol)
        self.action_space = Discrete(9)

    def set_agent_start(self, agent_start_state: int):
        x, y = HF.to_coords(agent_start_state, self.ncol)
        self.map[x][y] = 0

    def action_mapping(self, action: int, agent_state: int) -> int:
        hrow, hcol = row, col = HF.to_coords(agent_state, self.ncol)

        match action:  # Up + Left
            case 0:
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

            case 8:  # Down Right
                hcol += 1
                hrow += 1

        if 0 <= hcol <= self.ncol + 1 and 0 <= hrow <= self.nrow + 1:
            return HF.to_state((hrow, hcol), self.ncol)

        return agent_state  # No new state

    def calculate_reward(self):
        value_at_state = HF.get_loaction_value(
            self.map, HF.to_coords(self.agent_state, self.ncol)
        )
        reward = 0

        match value_at_state:
            case 1:  # Open Tile
                reward = 0.1 + self.EPISODE_LENGTH / 100

            case 2:  # Goal
                reward = 50

            case 3:  # Obstical
                reward = 0

        return reward

    def step(self, action: int) -> tuple[int, float, list, bool]:
        print(f"Action in env {action}")
        new_state = self.action_mapping(action, self.agent_state)
        reward = self.calculate_reward()
        info = []
        terminated = self.termination_check()
        self.EPISODE_LENGTH -= 1

        return new_state, reward, info, terminated

    def termination_check(self) -> bool:
        agent_coords = HF.to_coords(self.agent_state, self.ncol)
        value_at_state = HF.get_loaction_value(self.map, agent_coords)

        TERMINATION_CONDITIONS = [
            (value_at_state == 3),
            (value_at_state == 2),
            (self.EPISODE_LENGTH <= 0),
        ]

        if any(TERMINATION_CONDITIONS):
            return True

        return False

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
