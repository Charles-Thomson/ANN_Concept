import numpy as np

ENV_MAP = [
    [2, 2, 2, 2, 2, 2],
    [2, 1, 3, 1, 3, 2],
    [2, 1, 1, 1, 1, 2],
    [2, 1, 1, 2, 1, 2],
    [2, 1, 1, 2, 3, 2],
    [2, 2, 2, 2, 2, 2],
]


agent_path_holer = [7, 8, 10, 28]


def mapData(ENV_MAP: np.array):
    map = np.array(ENV_MAP)
    map = map.flatten()

    map_size = [len(ENV_MAP), len(ENV_MAP[0])]
    obstical_locations = np.where(map == 2)
    goal_locations = np.where(map == 3)

    print(obstical_locations)
    print(goal_locations)
    print(map_size)


if __name__ == "__main__":
    mapData(ENV_MAP)
