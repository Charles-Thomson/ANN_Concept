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
    [3, 1, 2, 1, 2],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]


class MazeEnv(Env):
    def __init__(self, agent_start_state: int, ENV_MAP: list = ENV_MAP):
        self.map = np.array(ENV_MAP)

        nrow, ncol = self.nrow, self.ncol = self.map.shape

        self.agent_start_state = agent_start_state

        self.agent_state = agent_start_state

        self.set_agent_start(agent_start_state)

        self.EPISODE_LENGTH = 3
        self.termination = False

        self.observation_space = Discrete(nrow * ncol)
        self.action_space = Discrete(9)

    def reset(self):
        self.agent_state = self.agent_start_state
        self.EPISODE_LENGTH = 50
        self.termination = False
        self.set_agent_start(self.agent_start_state)

        return self.agent_start_state

    def set_agent_start(self, agent_start_state: int):
        x, y = HF.to_coords(agent_start_state, self.ncol)
        self.map[x][y] = 0

    def action_mapping(self, action: int, agent_state: int) -> int:
        hrow, hcol = row, col = HF.to_coords(self.agent_state, self.ncol)
        self.invalid_move = False

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

            case 8:  # Down + Right
                hcol += 1
                hrow += 1

        if self.collision_check(hrow, hcol) is True:
            new_state = HF.to_state((hrow, hcol), self.ncol)
            self.agent_state = new_state
            return new_state

        self.invalid_move = True
        return self.agent_state

    def collision_check(self, x: int, y: int) -> bool:

        COLLISION_CHECKS = [
            (0 <= x <= self.ncol - 1),
            (0 <= y <= self.nrow - 1),
            (HF.to_state((x, y), self.ncol) != 2),
        ]

        # If the move is in bounds and not moving to an obstical
        if all(COLLISION_CHECKS):
            return True

        return False

    def calculate_reward(self):
        value_at_state = HF.get_loaction_value(
            self.map, HF.to_coords(self.agent_state, self.ncol)
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

    def step(self, action: int) -> tuple[int, float, list, bool]:
        self.action_mapping(action, self.agent_state)
        info = []
        new_state = self.agent_state
        reward = self.calculate_reward()
        self.termination_check()
        terminated = self.termination
        self.EPISODE_LENGTH -= 1

        if self.invalid_move is True:
            info = ["Move Invalid"]
            reward = 0

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
            self.termination = True

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
