import numpy as np
from math import sqrt


def to_npArray(ENV_MAP: list) -> np.array:
    map = np.array(ENV_MAP)
    map_dimension = int(sqrt(len(ENV_MAP)))
    map.reshape(map_dimension, map_dimension)
    return map


def to_state(coords: tuple, ncol: int) -> int:
    state = (coords[0] * ncol) + coords[1]
    return state


def to_coords(state: int, ncol: int) -> tuple:
    x = int(state / ncol)
    y = int(state % ncol)
    return (x, y)


def get_loaction_value(map: np.array, coords: tuple) -> int:
    x, y = coords
    value = map[x][y]
    return value


# Helper function - Simplify action to direcion equiverlent
def simple_move(action: int) -> str:
    simple_move = ""
    match action:
        case 0:
            simple_move = "Up + Left"

        case 1:
            simple_move = "Up"

        case 2:
            simple_move = "Up + Right"

        case 3:
            simple_move = "Left"

        case 4:
            simple_move = "No move"

        case 5:
            simple_move = "Right"

        case 6:
            simple_move = "Down + Left"

        case 7:
            simple_move = "Down"

        case 8:
            simple_move = "Down + Right"
    return simple_move

    # \\ --------------------------------------------Save episode to file \\
    def save_EPISODE(self, agent_path: list, reward: int, episode: int):
        path_length = str(len(agent_path))
        reward = str(reward)
        episode = str(episode)
        f = open("EpisodeData.txt", "a")
        f.write("\n" + episode)
        f.write(", " + path_length)
        f.write(", " + reward)
        f.close()
