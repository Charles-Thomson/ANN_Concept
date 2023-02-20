import numpy as np

episodes = 200
episode_length = 10

# Agent Brian
New_Generation_Threshold = 6

# Agent
fitness_threshold = 2.0
fitness_threshold_increase = 1.0
# Env
agent_start_state = 13


def mapData():
    map = np.array(ENV_MAP)
    map = map.flatten()

    map_size = [len(ENV_MAP), len(ENV_MAP[0])]
    obstical_locations = np.where(map == 2)
    obstical_locations = list(obstical_locations[0])

    goal_locations = np.where(map == 3)
    goal_locations = list(goal_locations[0])

    return map_size, obstical_locations, goal_locations


ENV_MAP = [
    [2, 2, 2, 1, 2, 2],
    [2, 1, 3, 2, 3, 2],
    [2, 1, 1, 1, 1, 2],
    [2, 1, 1, 2, 1, 2],
    [2, 1, 1, 2, 3, 2],
    [2, 2, 2, 2, 2, 2],
]

# ENV_MAP = [
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 3, 1, 3, 1],
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 2, 1, 1],
#     [1, 1, 1, 2, 3, 1],
#     [1, 1, 1, 1, 1, 1],
# ]

# 11 x 11
# agent_start_state = 26
# ENV_MAP = [
#     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#     [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 2],
#     [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2],
#     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
# ]
