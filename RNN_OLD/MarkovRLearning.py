"""
Markov decission prcess

The agent views the env state = St
The agent selects an action = At
Giving state action pair = (St, At)

Time is incremented
The env is now in a new state St+1
Agent gains reward of Rt+1 for the action At taken at env state St

Process = View env state, select action, time increments, gain reward
= S0 ,A0,R1 ,S1,A1 ,R2 ...
"""

import numpy as np

# import gym as gym
from gym import spaces

"""
Test map
0-> start
1-> open
2-> obstical
3-> goal
"""

TEST_MAP = [
    [0, 1, 2, 1, 3],
    [1, 1, 2, 1, 1],
    [2, 1, 2, 1, 2],
    [1, 1, 1, 1, 1],
    [1, 1, 2, 2, 1],
]


class MDP:
    def __init__(self, TEST_MAP: list):
        self.MAP = np.array(TEST_MAP)
        nrow, ncol = self.MAP.shape

        nS = nrow * ncol
        nA = 9

        # This creates a dict of each state with a dict nested inside
        # containing all the actions of that state
        self.p = {s: {a: [] for a in range(nA)} for s in range(nS)}

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        # Geth state from coords
        def to_state(row, col):
            state = row * ncol + col

            return state

        # Get coords from state
        def to_coords(state):
            x = int(state / ncol)
            y = int(state % ncol)

            return (x, y)

        def test_populate_p_matrix():
            for s in range(nS):
                row, col = to_coords(s)
                for a in range(nA):
                    li = self.p[s][a]

                    value = self.MAP[to_coords(s)]
                    if value in [2, 3]:
                        li.append((1.0, s, 0, True, a, s))

                    else:
                        # 1 / 8 as there are 8 moves inc 0
                        li.append(
                            (1.0 / 8.0, *update_probability_matrix(row, col, a), a, s)
                        )
                    print(li)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc_coords(row, col, action)
            newstate = to_state(newrow, newcol)
            newstatevalue = self.MAP[newrow, newcol]
            terminated = newstatevalue in [2, 3]  # check if obstical or goal
            reward = 0
            return newstate, reward, terminated

        def inc_coords(row, col, action):
            holder_row, holder_col = row, col

            match action:
                case 0:  # Up + Left
                    holder_col = col - 1
                    holder_row = row - 1

                case 1:  # Up
                    holder_row = row - 1

                case 2:  # Up + Right
                    holder_col = col + 1
                    holder_row = row - 1

                case 3:  # Left
                    holder_col = col - 1

                case 4:  # No move
                    pass

                case 5:  # Right
                    holder_col = col + 1

                case 6:  # Down + Left
                    holder_col = col - 1
                    holder_row = row + 1

                case 7:  # Down
                    holder_row = row + 1

                case 8:  # Down + Right
                    holder_col = col + 1
                    holder_row = row + 1

            # Guard for out of bounds
            if 0 <= holder_row <= nrow - 1 and 0 <= holder_col <= ncol - 1:
                return holder_row, holder_col

            return row, col

        def add_variance(a):
            for b in [
                (a - 1) % 4,
                a,
                (a + 1) % 4,
            ]:  # gives the varience to the action choice
                return b

        test_populate_p_matrix()


MDP(TEST_MAP)
